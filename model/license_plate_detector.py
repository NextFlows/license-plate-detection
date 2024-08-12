#!/usr/bin/env python
import torch
import cv2
import sys
from PIL import Image
import numpy as np
import os

import sys
sys.path.append('./src')

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords


class LicensePlateDetector:
    def __init__(self, weights: str, use_cuda: bool) -> None:
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.load_model(weights)

    def load_model(self, weights):
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

    def preprocess(self, img0, input_shape):
        img = letterbox(img0, input_shape, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def postprocess(self, pred, img, img0, conf_thres):
        pred = non_max_suppression(pred, conf_thres)
        for i, det in enumerate(pred): # detections per image
            if len(det):
                det = det.cpu()
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                pred[i] = det

        return pred[0].cpu().numpy()

    @torch.no_grad()
    def detect(
            self,
            image,
            conf_thres: float,
            input_shape: tuple
    ) -> tuple:
        """Function to detect license plate in an image
        Args:
            image (_type_): image on which to run detection
            conf_thres (float, optional): class conf threshold.
                                          Defaults to 0.5.
            input_shape (tuple, optional): input shape for model.
                                           Defaults to (1280, 1280).

        Returns:
            (tuple): returns bboxes (xyxy), scores, class_ids
        """

        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        img0 = image

        img = self.preprocess(img0, input_shape)
        pred = self.model(img)[0]
        result = self.postprocess(pred, img, img0, conf_thres)
        classes = ['license_plate']*len(result)
        return result[:, :4].astype(int).tolist(), result[:, 4].tolist(), classes

