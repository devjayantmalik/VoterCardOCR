import time

import cv2
import numpy as np
from PIL import Image
from doctr.models import ocr_predictor
import torch


import requests

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
        return None


if __name__ == '__main__':
   # load model
    model = ocr_predictor('db_resnet50', 'parseq',
                          pretrained=True).cuda() if torch.cuda.is_available() else ocr_predictor('db_resnet50',
                                                                                                  'parseq',
                                                                                                  pretrained=True)
    # Regularly keep running and solving tasks
    while True:
        data = fetch_results_from_api()
        if len(data) == 0:
            time.sleep(10) # sleep 10 seconds
            continue

        # start processing data






    def clean_and_preprocess(box: np.ndarray):
        cropped = Image.fromarray(box)
        cropped = cropped.resize((cropped.size[0] * 4, cropped.size[1] * 4))
        cropped = cropped.crop((cropped.width / 2, 0, cropped.width, 140))
        cropped = cropped.convert("RGB")
        return np.asarray(cropped)

    # Crop all voter card numbers from slips
    start = time.time()
    card_nos = [clean_and_preprocess(pic) for pic in detector.detect(pil_picture=picture)]
    print(f"Clean and Preprocess completed in {time.time() - start} seconds")

    start = time.time()
    predictions = model(card_nos)
    print(f"Prediction of {len(card_nos)} items completed in {time.time() - start} seconds")
    print(predictions.export())

    # for i, cropped in enumerate():
    #     start = time.time()
    #     out = model([])
    #     print(f"Model Prediction in {time.time() - start} seconds")
    #     data = [(word.value, word.confidence) for line in out.pages[0].blocks[0].lines for word in line.words]
    #     cropped.save(f"pics/{i}__{data[0][0].replace('/', '_')}.png")
    #     print(data)

        # cropped = cv2.resize(cropped, (cropped.shape[1] * 4, cropped.shape[0] * 4))
        # cv2.imwrite("cropped.png", cropped)
        # cv2.imshow("preview", cropped)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
