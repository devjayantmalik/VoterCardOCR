import os
import sys
import time

import cv2
import numpy as np
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import Document 
import torch

import requests
import base64
import io
import json
from nanoid import generate


def fetch_results_from_api(device_id):
    api_url = f"http://localhost:8000/get-my-tasks/{device_id}"  # Replace with your actual API URL
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        tasks = response.json()
        return tasks
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return [] 


def parse_model_output(output):
    words = output.blocks[0].lines[0].words

    prediction = words[0].value.strip()
    export = json.dumps(output.export())
    confidence = words[0].confidence if (len(words) == 1) else 0
    errors = "More than 1 word found" if (len(words) > 1) else "Length neither 10 nor 16" if (
            len(prediction) != 10 or len(prediction) != 16) else ""
    return prediction, export, confidence, errors


def post_api_data_to_server(device_id: int, data):
    api_url = f"http://localhost:8000/submit-results/{device_id}"  # Replace with your actual API URL
    headers = {
        'Content-Type': 'application/json',
        # Add any additional headers if needed
    }

    try:
        response = requests.post(api_url, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error posting data to API: {e}")
        return False


if __name__ == '__main__':
    device_id = None
    try:
        if len(sys.argv) < 2:
            raise ValueError("Device id is missing")
        device_id = int(sys.argv[1])
    except ValueError:
        print("Device id is required and must be passed, when calling script. Example: python3 main.py 1")
        exit(0)

    # load model
    model = ocr_predictor('db_resnet50', 'parseq',
                          pretrained=True).cuda() if torch.cuda.is_available() else ocr_predictor('db_resnet50',
                                                                                                  'parseq',
                                                                                                  pretrained=True)

    # Regularly keep running and solving tasks
    while True:
        # make request for old processed data incase those were unsuccessful
        os.makedirs("data/predictions", exist_ok=True)
        files = [os.path.join("data/predictions", filename) for filename in os.listdir("data/predictions")]
        for filepath in files:
            with open(filepath, 'r') as file:
                api_data = json.loads(file.read())
                success = post_api_data_to_server(device_id, api_data)
                if success:
                    os.remove(filepath)

        data = fetch_results_from_api(device_id)
        print(f"Fetched {len(data)} items from network")
        if data is None or len(data) == 0:
            print("sleeping for 10 seconds")
            time.sleep(5)  # sleep 10 seconds
            continue

        # convert base64 to bytesio and pillow image
        decoded = [io.BytesIO(base64.b64decode(pic["picture_base64"])) for pic in data]
        inputs = [np.asarray(Image.open(item)) for item in decoded]

        # perform predictions
        start = time.time()
        predictions = model(inputs)
        print(f"Prediction of {len(inputs)} items completed in {time.time() - start} seconds")

        api_data = []
        for i in range(len(inputs)):
            data_item, output = data[i], predictions.pages[i]

            prediction, export, confidence, errors = parse_model_output(output)
            api_data.append({
                "id": data_item['id'],
                "prediction": prediction,
                "prediction_export": export,
                "confidence": confidence,
                "errors": errors,
            })

        # make get request to server for results
        success = post_api_data_to_server(device_id, api_data)

        # save to predictions directory
        if not success:
            filename = generate("abcdefghijklmnopqrstuvwxyz", 15) + ".json"
            with open(os.path.join("data", "predictions", filename), 'w') as file:
                file.write(json.dumps(api_data))
