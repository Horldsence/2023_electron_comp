import cv2
from basicFunc.picam import Imget
import numpy as np
from procFunc.imageProc import imageProcessor
from procFunc.CannyProc import RectangleDetector
from procFunc.mathAssoc import mathProc
import torch
from basicFunc.yoloLibrary import DetectMultiBackend,LoadImages,Annotator,non_max_suppression,scale_coords,colors
import cv2
from basicFunc.picam import Imget
import numpy as np
import base64

# yolo模型加载
device = torch.device('cpu')
model = DetectMultiBackend("yoloModel/best.pt", device=device, dnn=False)
if model.pt:
    model.model.float()
print("模型加载完成")
def detect_img(img0): #预测
    device = torch.device('cpu')
    stride, names = model.stride, model.names  #读取模型的步长和模型的识别类别
    dataset = LoadImages(img0=img0, img_size=[640, 640], stride=stride, auto=False)  #对读取的图片进行格式化处理
    # print(dataset)
    for im, im0s in dataset:
        im = (torch.from_numpy(im).to(device).float()/255)[None]  #把图片数据转换成张量
        pred = model(im, augment=False, visualize=False)   #进行检测
        det = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)[0]  #对检测结果进行处理

        im0 = im0s.copy()
        annotator = Annotator(im0, line_width=0, example=str(names))

        data = dict.fromkeys(names, 0)
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  #识别结果都在这里了

            for *xyxy, conf, cls in reversed(det):   #xyxy是识别结果的标注框的坐标，conf是识别结果的置信度，cls是识别结果对应的类别
                c = int(cls)
                if names[c] in data:
                    data[names[c]] += 1
                else:
                    data[names[c]] = 1
                label = (f'{names[c][0]}{data[names[c]]}')
                annotator.box_label(xyxy, label, color=colors(c, True))  #对图片进行标注，就是画框

        im0 = annotator.result()
        return data, im0

getImg = Imget()
imProc = imageProcessor()
mtProc = mathProc()
rectangleFinder = RectangleDetector(3, 5)

if __name__ == "__main__":
    redPoint = (0, 0)
    greenPoint = (0, 0)
    Point = (0, 0)
    previousRedPoint = (0, 0)
    while True:
        img = getImg.getImg()
        originalImg = img.copy()

        # 数据获取及图像预处理
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc_image, green_points, red_points = imProc.find_small_points(img)
        doted_image, point_list = imProc.find_bright_spots(img)
        red_point_calc = mtProc.calculate_centroid(red_points)
        green_point_calc = mtProc.calculate_centroid(green_points)
        cv2.imshow("original_image", img)
        # cv2.imshow("new_img", proc_image)
        # cv2.imshow("gray_image", gray_image)
        count, im0 = detect_img(img)
        cv2.imshow("doted_image", doted_image)
        # cv2.imshow("predict image", img_prediction)

        # 寻找长方形
        rectangle_img, rectangles = rectangleFinder.find_rectangles(originalImg)
        cv2.imshow("rectangle_img", rectangle_img)
        print(rectangles)
        cv2.waitKey(3)
