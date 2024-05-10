import cv2
from basicFunc.picam import Imget
import numpy as np
from procFunc.imageProc import imageProcessor
from procFunc.CannyProc import RectangleDetector
from procFunc.mathAssoc import mathProc
from procFunc.yoloProc import yolov5Detector

getImg = Imget()
imProc = imageProcessor()
mtProc = mathProc()
rectangleFinder = RectangleDetector(3, 5)
detector = yolov5Detector('./yoloModel/best.pt')

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
        cv2.imshow("doted_image", doted_image)
        result = detector.detect(img)
        img_prediction = detector.drawPredictions(img, result)
        cv2.imshow("predict image", img_prediction)

        # # 红绿点识别区分
        # point_num = len(point_list)
        # try:
        #     Point = [int(x) for x in (point_list[0])[:2]]
        # except IndexError:
        #     Point = (0, 0)
        # if red_point_calc != () and green_point_calc != ():
        #     for Point in point_list:
        #         if mtProc.calculate_distance(Point, red_point_calc) <= mtProc.calculate_distance(Point, green_point_calc) and mtProc.calculate_distance(Point, red_point_calc) <= 20:
        #             redPoint = Point
        #         else:
        #             greenPoint = Point
        # else:
        #     if len(point_list) == 0:
        #         if red_point_calc != () and red_point_calc != (0, 0):
        #             redPoint = [int(x) for x in red_point_calc]
        #         else:
        #             redPoint = previousRedPoint
        #     else:
        #         redPoint = Point
        # try:
        #     cv2.circle(img, redPoint, 25, (0, 0, 255), 3)
        #     cv2.circle(img, greenPoint, 25, (0, 255, 0), 3)
        # except cv2.error:
        #     print("there is no point")
        # previousRedPoint = redPoint
        # cv2.imshow("result_image", img)
        # print("Green Points (x, y):", greenPoint)
        # print("Red Points (x, y):", redPoint)

        # 寻找长方形
        rectangle_img, rectangles = rectangleFinder.find_rectangles(originalImg)
        cv2.imshow("rectangle_img", rectangle_img)
        print(rectangles)
        cv2.waitKey(3)
