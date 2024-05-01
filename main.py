import cv2
from picam import Imget
import numpy as np
from imageProc import imageProcessor
from mathAssoc import mathProc

getImg = Imget()
imProc = imageProcessor()
mtProc = mathProc()
redPoint = ()

while True:
    img = getImg.getImg()
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 请将“your_image_path.jpg”替换成实际图像的路径
    proc_image, green_points, red_points = imProc.find_small_points(img)
    doted_image, point_list = imProc.find_bright_points(img)
    cv2.imshow("original_image", img)
    cv2.imshow("new_img", proc_image)
    cv2.imshow("gray_image", gray_image)
    cv2.imshow("gray_image", doted_image)
    cv2.waitKey(1)
    for Point in point_list:
        for PointR in red_points:
            for PointG in green_points:
                if mtProc.calculate_distance(Point, PointG) and mtProc.calculate_distance(Point, PointR):
                    redPoint = Point
    print("Green Points (x, y, diameter):", green_points)
    print("Red Points (x, y, diameter):", red_points)