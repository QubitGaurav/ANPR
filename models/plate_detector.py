import os
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path='runs/license_plate_detector/weights/best.pt'):
        """
        Initialize the plate detector with the custom trained YOLO model.
        Falls back to base YOLOv8 model if the custom weights are not found.
        """
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"Warning: Custom model not found at {model_path}, falling back to base YOLO.")
            self.model = YOLO('yolov8n.pt')
            
    def detect(self, image):
        """
        Detect license plates in an image.
        Returns a list of dictionaries with bounding box and confidence.
        """
        # Run inference (we expect the model to predict class 0 for license plate)
        results = self.model(image, verbose=False)
        
        plates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Assuming class 0 is license plate. If we fallback to base yolo, 
                # we just return all boxes. In a real system, the fallback is just a placeholder.
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                plates.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf
                })
                
        return plates
