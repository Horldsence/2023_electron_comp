import torch
from torchvision.ops import nms
from PIL import Image, ImageDraw, ImageFont

class yolov5Detector:
    def __init__(self, weights_path: str, device: str = 'cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', path=weights_path).to(device)
        self.model.eval()

    def detect(self, image, conf_thres=0.25, iou_thres=0.45):
        img = torch.zeros((1, 3, 640, 640), device = self.device)
        img_raw = torch.from_numpy(image)
        if len(img_raw.shape) == 2:
            img_raw = img_raw.unsqueeze(-1).repeat(1, 1, 3)
        img_raw = img_raw.premute(2, 0, 1)
        img_raw = img_raw.to(self.device).float() / 255.0 #图片归一化
        img[0, :, :img_raw.shape[1], :img_raw.shape[2]] = img_raw

        result = self.model(img, size=640)

        # NMS 处理
        pred = result.pred[0]
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = pred[:, 5]
        indices = nms(boxes, scores, iou_thres)

        detections = []
        for ind in indices:
            bbox = boxes[ind].tolist()
            score = scores[ind].item()
            class_id = int(classes[ind].item())
            detections.append((class_id, score, bbox))

        return detections

    def drawPredictions(self, image, detections, line_thickness=3):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        for det in detections:
            class_id, confidence, bbox = det
            x1, y1, x2, y2 = bbox
            label = f"{class_id}({confidence:.2f})"
            draw.rectangle([x1, y1, x2, y2], outline='red',width=line_thickness)
            draw.text((x1, y1), label, fill="white", font=font)

        return image