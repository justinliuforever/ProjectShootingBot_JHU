import cv2
import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class TargetDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the target detector
        Args:
            model_path: Path to the YOLO model
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model.to(self.device)
        
        # 缓存常用的颜色和字体设置
        self.BOX_COLOR = (0, 255, 0)  # BGR格式
        self.TEXT_COLOR = (0, 255, 0)
        self.LINE_THICKNESS = 2
        self.FONT_SCALE = 0.9
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

    def load_model(self, model_path: str) -> YOLO:
        """
        Load the YOLO model
        Args:
            model_path: Path to the model file
        Returns:
            YOLO model instance
        """
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        Detect targets in the image
        Args:
            image: Input image as numpy array (BGR format)
        Returns:
            List of bounding boxes in [x1,y1,x2,y2] format
        """
        if image is None or image.size == 0:
            return []

        try:
            with torch.no_grad():
                results = self.model(image, 
                                   conf=self.conf_threshold, 
                                   iou=self.iou_threshold, 
                                   device=self.device)
                
                boxes = []
                if len(results) > 0:
                    for r in results:
                        boxes_tensor = r.boxes.cpu().numpy()
                        for box in boxes_tensor:
                            coords = box.xyxy[0].tolist()
                            boxes.append(coords)
                
                return boxes

        except Exception as e:
            print(f"Detection error: {str(e)}")
            return []

    def draw_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        """
        Draw detection boxes on the image
        Args:
            image: Input image
            boxes: List of bounding boxes
        Returns:
            Image with drawn boxes
        """
        if image is None or len(boxes) == 0:
            return image

        image_with_boxes = image.copy()
        
        for box in boxes:
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(image_with_boxes, 
                         (x1, y1), 
                         (x2, y2), 
                         self.BOX_COLOR, 
                         self.LINE_THICKNESS)
            
            # Add label
            cv2.putText(image_with_boxes, 
                       'Target', 
                       (x1, y1 - 10), 
                       self.FONT, 
                       self.FONT_SCALE, 
                       self.TEXT_COLOR, 
                       self.LINE_THICKNESS)
        
        return image_with_boxes

def process_image(image_path: str, detector: TargetDetector) -> np.ndarray:
    """
    Process a single image file
    Args:
        image_path: Path to the image file
        detector: TargetDetector instance
    Returns:
        Processed image with detection boxes
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        boxes = detector.detect(image)
        image_with_boxes = detector.draw_boxes(image, boxes)
        return image_with_boxes
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None
