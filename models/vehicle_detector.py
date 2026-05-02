import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize the vehicle detector with a base YOLO model.
        """
        # Load the base YOLO model (downloads automatically if not present)
        self.model = YOLO(model_path)
    
    def detect(self, image):
        """
        Detect vehicles in an image.
        Returns a list of dictionaries with bounding box and confidence.
        """
        # COCO class IDs: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
        results = self.model(image, classes=[2, 3, 5, 7], verbose=False)
        
        vehicles = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                vehicles.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
                
        return vehicles
