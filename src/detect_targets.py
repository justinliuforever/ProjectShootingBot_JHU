import cv2
import torch
from ultralytics import YOLO
import numpy as np

class TargetDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model.to(self.device)

    def load_model(self, model_path):
        model = YOLO(model_path)
        return model

    def detect(self, image):
        with torch.no_grad():
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
        return results

    def draw_boxes(self, image, results):
        for r in results:
            boxes = r.boxes.cpu().numpy()
            for box in boxes:
                b = box.xyxy[0]
                c = int(box.cls)
                conf = float(box.conf)
                cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(image, f'{r.names[c]} {conf:.2f}', (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

def process_image(image_path, detector):
    image = cv2.imread(image_path)
    results = detector.detect(image)
    image_with_boxes = detector.draw_boxes(image, results)
    return image_with_boxes
