from fastapi import FastAPI, Path, Request
import sqlite3
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from pydantic import BaseModel

# SQLite connection setup
conn = sqlite3.connect('data/database.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

limiter = Limiter(key_func=get_remote_address, enabled=True)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/get-my-tasks/{device_id}")
@limiter.limit("30/minute")
async def get_my_tasks(request: Request, device_id: int = Path(..., description="The device ID")):
    # Check if device_id exists in devices table
    cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
    device = cursor.fetchone()
    if not device:
        return []  # Device not found, return empty list as response

    # Assign device_id to up to 1000 rows in jobs table where device_id is NULL
    cursor.execute("""
        UPDATE jobs
        SET device_id = ?
        WHERE device_id IS NULL
        LIMIT 50
    """, (device_id,))
    conn.commit()

    # Fetch tasks for the device where is_task_complete is False
    cursor.execute("""
        SELECT * FROM jobs
        WHERE device_id = ? AND is_task_complete = FALSE
        LIMIT 50
    """, (device_id,))
    tasks = cursor.fetchall()

    return tasks


class TaskResult(BaseModel):
    id: int
    prediction: str
    prediction_export: str
    confidence: float
    errors: str


@app.post("/submit-results/{device_id}")
@limiter.limit("100/minute")
async def submit_results(device_id: int, results: list[TaskResult], request: Request):
    # Check if device_id exists in devices table
    cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
    device = cursor.fetchone()
    if not device:
        return {"message": f"Results failed to submit for device {device_id} as it doesn't exist."}

    for result in results:
        job_id = result.id
        prediction = result.prediction
        prediction_export = result.prediction_export
        confidence = result.confidence
        errors = result.errors

        # Update job in database
        cursor.execute("""
            UPDATE jobs
            SET prediction = ?,
                prediction_export = ?,
                confidence = ?,
                errors = ?,
                is_task_complete = TRUE,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND device_id = ?
        """, (prediction, prediction_export, confidence, errors, job_id, device_id))

    conn.commit()

    return {"message": f"Results submitted successfully for device {device_id}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
