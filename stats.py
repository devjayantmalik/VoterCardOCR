from datetime import datetime, timedelta
import sqlite3


# SQLite connection setup
conn = sqlite3.connect('data/database.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()


def count_uncompleted_tasks_last_hour():
    # Calculate the datetime 1 hour ago from now
    one_hour_ago = datetime.now() - timedelta(hours=1)

    # Query to count tasks assigned but not completed within the last 1 hour
    cursor.execute("""
        SELECT COUNT(*) AS num_tasks
        FROM jobs
        WHERE is_task_complete = FALSE
          AND updated_at <= ?
    """, (one_hour_ago,))

    row = cursor.fetchone()
    if row:
        return row['num_tasks']
    else:
        return 0


if __name__ == '__main__':
    num_tasks = count_uncompleted_tasks_last_hour()
    print(f"No of tasks assigned but not completed in last 1 hour: {num_tasks}")