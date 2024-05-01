import cv2
import numpy as np

# 定义红色和绿色在HSV色彩空间的范围
lower_green = np.array([40, 70, 50])
upper_green = np.array([90, 255, 255])
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

class imageProcessor:
    def __init__(self) -> None:
        pass

    def calculate_aspect_ratio(self, w, h):
        return float(w) / h if h > 0 else 0

    def find_small_points(self, image, max_aspect_ratio = 1.5):
        # 读取和转换图像
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
            aspect_ratio = self.calculate_aspect_ratio(w, h)
            diameter = max(w, h)
            if diameter >= 5 and diameter <= 15 and aspect_ratio <= max_aspect_ratio:
                small_green_points.append((x, y, diameter))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Bounding box for green point

        for contour in red_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = self.calculate_aspect_ratio(w, h)
            diameter = max(w, h)
            if diameter >= 5 and diameter <= 15 and aspect_ratio <= max_aspect_ratio:
                small_red_points.append((x, y, diameter))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)  # Bounding box for red point

        return image, small_green_points, small_red_points

    def find_bright_spots(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用阈值，只保留亮度在220到255范围内的亮点
        _, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # 形态学操作
        thresh = cv2.erode(thresholded, None, iterations=2)
        thresholded = cv2.dilate(thresh, None, iterations=4)

        # 找到亮点的轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers = []  # 存储中心坐标的列表
        for contour in contours:
            # 计算轮廓的矩
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # 计算中心坐标
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
                # 在原图上标注中心点
                cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
        # 返回标注后的图像和中心点坐标列表
        return image, centers