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