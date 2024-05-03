import cv2
import numpy as np

class calcProc:
    def __init__(self) -> None:
        pass
    def non_max_suppression(self, gradient_magnitude, gradient_direction):
        """应用非极大值抑制至梯度幅度图。

        :param gradient_magnitude: 梯度幅度图
        :param gradient_direction: 梯度方向图
        :return: 经过非极大值抑制的边缘图
        """
        # 获取图像维度
        M, N = gradient_magnitude.shape
        # 初始化输出图像
        output = np.zeros_like(gradient_magnitude)

        # 遍历图像中的每个像素（除开边缘，边缘通常没有足够的邻居）
        for i in range(1, M-1):
            for j in range(1, N-1):
                # 获取当前像素点的梯度幅度和方向
                mag = gradient_magnitude[i, j]
                direction = gradient_direction[i, j]

                # 根据梯度方向找到前后两个邻居的坐标
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                    neighbour_1 = gradient_magnitude[i, j+1]
                    neighbour_2 = gradient_magnitude[i, j-1]
                elif (22.5 <= direction < 67.5):
                    neighbour_1 = gradient_magnitude[i+1, j-1]
                    neighbour_2 = gradient_magnitude[i-1, j+1]
                elif (67.5 <= direction < 112.5):
                    neighbour_1 = gradient_magnitude[i+1, j]
                    neighbour_2 = gradient_magnitude[i-1, j]
                elif (112.5 <= direction < 157.5):
                    neighbour_1 = gradient_magnitude[i-1, j-1]
                    neighbour_2 = gradient_magnitude[i+1, j+1]

                # 对当前像素点施加非极大值抑制
                if mag >= neighbour_1 and mag >= neighbour_2:
                    output[i, j] = mag
        return output

    def get_magnitude_direction(self, image):
        # 使用Sobel算子计算梯度幅度和方向
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_direction = np.arctan2(sobely, sobelx) * (180 / np.pi) % 180

        return gradient_magnitude, gradient_direction

    def nms(self, img):
        gradient_magnitude, gradient_direction = self.get_magnitude_direction(img)
        return self.non_max_suppression(gradient_magnitude, gradient_direction)

class RectangleDetector:
    def __init__(self, minOpen, minClose):
        self.minOpen = minOpen  # 设置宽度过滤的最小值
        self.minClose = minClose

    def find_rectangles(self, img):
        rectangles = []
        imgContour = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 创建一个横向结构元素
        kernelOpen = cv2.getStructuringElement(cv2.MORPH_RECT, (self.minOpen, self.minOpen))
        # 进行开运算以去除结构
        imgGray = cv2.morphologyEx(imgGray, cv2.MORPH_OPEN, kernelOpen, iterations=1)

        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # 高斯模糊
        imgCanny = cv2.Canny(imgBlur, 60, 60)  # Canny算子边缘检测
        kernelClose = cv2.getStructuringElement(cv2.MORPH_RECT, (self.minClose, self.minClose))
        imgCanny = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, kernelClose, iterations=1)

        cv2.imshow("imgCanny", imgCanny)
        # cv2.imshow("imgMorph", imgMorph)  # 显示形态学处理后的图像

        # 寻找轮廓点
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for obj in contours:
            area = cv2.contourArea(obj)  # 计算轮廓内区域的面积
            cv2.drawContours(imgContour, obj, -1, (255, 0, 0), 4)  # 绘制轮廓线
            perimeter = cv2.arcLength(obj, True)  # 计算轮廓周长
            approx = cv2.approxPolyDP(obj, 0.02 * perimeter, True)  # 获取轮廓角点坐标
            CornerNum = len(approx)  # 轮廓角点的数量
            x, y, w, h = cv2.boundingRect(approx)  # 获取坐标值和宽度、高度

            # 轮廓对象分类
            if area > 10 and CornerNum == 4 and w != h:
                # 绘制边界框
                cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 0, 255), 2)
                rectangles.append(approx)

        return imgContour, rectangles