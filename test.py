from picam import Imget
import cv2
import numpy as np

def find_bright_spots(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值，只保留亮度在220到255范围内的亮点
    _, thresholded = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

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
getImg = Imget()
while True:
  image = getImg.getImg()
  # 使用函数
  marked_image, bright_spots_centers = find_bright_spots("your_image.jpg")

  # 显示图像
  cv2.imshow("Marked Image", marked_image)
  cv2.waitKey(1)

  # 打印中心坐标列表
  print("Bright spots centers:", bright_spots_centers)