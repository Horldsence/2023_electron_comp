import cv2
from basicFunc.picam import Imget
import numpy as np
from procFunc.imageProc import imageProcessor
from procFunc.CannyProc import RectangleDetector
from procFunc.mathAssoc import mathProc
from procFunc.yolov5Proc import detect_img

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
        cv2.imshow("original_image", img)
        count, im0, centerPointList = detect_img(img)
        cv2.imshow("doted_image", im0)
        print(centerPointList)

        # 寻找长方形
        rectangle_img, rectangles = rectangleFinder.find_rectangles(originalImg)
        cv2.imshow("rectangle_img", rectangle_img)
        print(rectangles)
        cv2.waitKey(3)
