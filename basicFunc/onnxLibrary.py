import cv2
import numpy as np
import onnxruntime
from pycocotools import mask as maskUtils

class ObjectDetector:
    def __init__(self, onnx_model):
        self.session = onnxruntime.InferenceSession(onnx_model)
        self.input_name = self.session.get_inputs()[0].name

    def run_onnx_model(self, image):
        # Resize the image to 320x320
        image = cv2.resize(image, (320, 320))
        # Convert the image to float
        image = image.astype(np.float32)
        raw_result = self.session.run(None, {self.input_name: image})
        return raw_result

    def apply_nms(self, raw_result, iou_threshold=0.5):
        boxes, scores, classes = raw_result
        indices = maskUtils.non_max_suppression(np.array(boxes), np.array(scores), iou_threshold)
        return boxes[indices], scores[indices], classes[indices]

    def draw_boxes_and_centers(self, image, boxes, classes):
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = box
            center = [(x1+x2)/2, (y1+y2)/2]
            centers.append(center)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, tuple(map(int, center)), 5, (0, 0, 255), -1)
        return image, centers