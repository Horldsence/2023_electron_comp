import cv2
import numpy as np
import measure
import imutils

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

    def find_blink_points(self, image, min_area = 50):
        # 颜色转化模糊处理
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurredImg = cv2.GaussianBlur(grayImg, (11, 11), 0)
        thresh = cv2.threshold(blurredImg, 230, 255, cv2.THRESH_BINARY)[1]
        # 膨胀腐蚀去除小斑点
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=4)
        # 连接组件分析
        labels = measure.label(thresh, neighbors=8, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        for label in np.unique(labels):
        # if this is the background label, ignore it
            if label == 0:
                continue
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            if numPixels > 300:
                mask = cv2.add(mask, labelMask)
        cv2.imshow('mask',mask)
        cv2.waitKey(0)
        # 查找亮点的轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 筛选出具有较小面积的亮点轮廓
        bright_spots = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius),
            (0, 0, 255), 3)