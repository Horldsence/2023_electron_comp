import cv2
from picam import Imget
import numpy as np
from imageProc import imageProcessor

getImg = Imget()
imProc = imageProcessor()

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
    for (brightX, brightY) in point_list:
        for (Rx, Ry) in red_points:
            pass
        for (Gx, Gy) in green_points:
            pass
    print("Green Points (x, y, diameter):", green_points)
    print("Red Points (x, y, diameter):", red_points)