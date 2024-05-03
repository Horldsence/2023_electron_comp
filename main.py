import cv2
from picam import Imget
import numpy as np
from imageProc import imageProcessor
from CannyProc import RectangleDetector
from mathAssoc import mathProc

getImg = Imget()
imProc = imageProcessor()
mtProc = mathProc()
rectangleFinder = RectangleDetector(3, 5)

if __name__ == "__main__":
    redPoint = (0, 0)
    greenPoint = (0, 0)
    while True:
        img = getImg.getImg()
        originalImg = img.copy()
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc_image, green_points, red_points = imProc.find_small_points(img)
        doted_image, point_list = imProc.find_bright_spots(img)
        red_point_calc = mtProc.calculate_centroid(red_points)
        green_point_calc = mtProc.calculate_centroid(green_points)
        cv2.imshow("original_image", img)
        cv2.imshow("new_img", proc_image)
        cv2.imshow("gray_image", gray_image)
        cv2.imshow("gray_image", doted_image)
        point_num = len(point_list)
        for Point in point_list:
            if red_point_calc != () and green_point_calc != ():
                if mtProc.calculate_distance(Point, red_point_calc) <= mtProc.calculate_distance(Point, green_point_calc) and mtProc.calculate_distance(Point, red_point_calc) <= 20:
                    redPoint = Point
                else:
                    greenPoint = Point
            else:
                if len(point_list) == 0:
                    try:
                        redPoint = red_point_calc
                    except IndexError:
                        redPoint = Point
                        print("redPoint Not Find")
                else:
                    redPoint = Point
        cv2.circle(img, redPoint, 25, (256, 0, 0), 3)
        cv2.imshow("result_image", img)
        print("Green Points (x, y):", greenPoint)
        print("Red Points (x, y):", redPoint)
        rectangle_img, rectangles = rectangleFinder.find_rectangles(originalImg)
        cv2.imshow("rectangle_img", rectangle_img)
        print(rectangles)
        cv2.waitKey(3)