from picam import Imget
import cv2
import numpy as np

getImg = Imget()
while True:
    image = getImg.getImg()
    # 使用函数
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 显示图像
    cv2.imshow("Marked Image", hsv_image)
    cv2.waitKey(1)