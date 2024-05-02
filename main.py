import cv2
from picam import Imget
import numpy as np
from imageProc import imageProcessor
from mathAssoc import mathProc

getImg = Imget()
imProc = imageProcessor()
mtProc = mathProc()

if __name__ == "__main__":
    while True:
        img = getImg.getImg()
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proc_image, green_points, red_points = imProc.find_small_points(img)
        doted_image, point_list = imProc.find_bright_spots(img)
        red_point_calc = mtProc.calculate_centroid(red_points)
        green_point_calc = mtProc.calculate_centroid(green_points)
        cv2.imshow("original_image", img)
        cv2.imshow("new_img", proc_image)
        cv2.imshow("gray_image", gray_image)
        cv2.imshow("gray_image", doted_image)
        for Point in point_list:
            if red_point_calc != () and green_point_calc != ():
                if mtProc.calculate_distance(Point, red_point_calc) <= mtProc.calculate_distance(Point, green_point_calc):
                    redPoint = Point
                    cv2.circle(img, redPoint, 25, (256, 0, 0), 3)
                else:
                    greenPoint = Point
                    cv2.circle(img, greenPoint, 25, (0, 256, 0), 3)
            else:
                redPoint = Point
                cv2.circle(img, redPoint, 25, (256, 0, 0), 3)
        cv2.imshow("result_image", img)
        print("Green Points (x, y, diameter):", green_points)
        print("Red Points (x, y, diameter):", red_points)
        rectangle_img, rectangles = imProc.find_rectangles(img)
        cv2.imshow("rectangle_img", doted_image)
        print(rectangles)
        cv2.waitKey(1)