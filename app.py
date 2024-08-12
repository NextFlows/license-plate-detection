import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse


import io
import cv2
import json
import time
import base64
import requests
import logging
from PIL import Image, ImageFilter
import numpy as np
import sys

from model.license_plate_detector import LicensePlateDetector

APP = FastAPI()
DEFAULT_CONF_THRESH = 0.8
model = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ===============================================================
# APP
# ===============================================================

@APP.on_event('startup')
def init():
    logging.info('Loading model...')
    global model
    model = LicensePlateDetector(
                    weights='trained_weights/license-plate-detection.pt',
                    use_cuda=False
    )
    logging.info('Model loaded successfully!')

@APP.post('/detect_license_plate')
async def detect_license_plate(file: UploadFile = File(...)):
    tic = time.time()
    image = Image.open(file.file)

    bboxes, scores, cls = model.detect(image=image,
                                       conf_thres=DEFAULT_CONF_THRESH,
                                       input_shape=(640, 640))

    # Convert image to np array and show bounding box on it
    image = np.array(image)

    # Draw on image
    for (x1,y1,x2,y2), score, cl in zip(bboxes, scores, cls):
        if cl == 'license_plate':
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0,0,255), thickness=2)

    _, image = cv2.imencode('.png', np.array(image))
    return StreamingResponse(io.BytesIO(image.tobytes()), media_type='image/png')


APP.mount('/', StaticFiles(directory='static', html=True), name='static')

@APP.get('/')
def index() -> FileResponse:
    return FileResponse(path='/app/static/index.html', media_type='text/html')
