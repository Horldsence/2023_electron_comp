import cv2
import 

import cv2
import numpy as np

# 定义红色和绿色在HSV色彩空间的范围
lower_green = np.array([40, 70, 50])
upper_green = np.array([90, 255, 255])
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

def find_small_points(image_path):
    # 读取和转换图像
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 在HSV空间里面根据定义的范围查找绿色和红色
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # 找到轮廓
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤直径在0-10像素的小点
    small_green_points = []
    small_red_points = []
    for contour in green_contours:
        # 计算轮廓的边界矩形，以估计直径
        (x, y, w, h) = cv2.boundingRect(contour)
        diameter = max(w, h)
        if diameter <= 10:
            small_green_points.append((x, y, diameter))

    for contour in red_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        diameter = max(w, h)
        if diameter <= 10:
            small_red_points.append((x, y, diameter))

    return small_green_points, small_red_points

# 请将“your_image_path.jpg”替换成实际图像的路径
green_points, red_points = find_small_points("your_image_path.jpg")
print("Green Points (x, y, diameter):", green_points)
print("Red Points (x, y, diameter):", red_points)