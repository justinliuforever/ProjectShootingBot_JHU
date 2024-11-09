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
        """
        Detect targets and return both YOLO results and simplified box coordinates
        Returns:
            tuple: (YOLO results, list of [x1,y1,x2,y2] coordinates)
        """
        with torch.no_grad():
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
            
            # Extract box coordinates for easier processing
            boxes = []
            if len(results) > 0:
                for r in results:
                    boxes_tensor = r.boxes.cpu().numpy()
                    for box in boxes_tensor:
                        # Get box coordinates [x1,y1,x2,y2]
                        coords = box.xyxy[0].tolist()
                        boxes.append(coords)
            
            return boxes  # Return just the coordinate list

    def draw_boxes(self, image, boxes):
        """Draw boxes on image using simplified coordinates"""
        for box in boxes:
            # Draw rectangle
            cv2.rectangle(image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 255, 0), 2)
            
            # Add label (simplified to just show "Target")
            cv2.putText(image, 
                       'Target', 
                       (int(box[0]), int(box[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, 
                       (0, 255, 0), 
                       2)
        return image

def process_image(image_path, detector):
    image = cv2.imread(image_path)
    boxes = detector.detect(image)
    image_with_boxes = detector.draw_boxes(image, boxes)
    return image_with_boxes
