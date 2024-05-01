import cv2
import math
import numpy as np

class mathProc:
    def __init__(self) -> None:
        pass

    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    def calculate_centroid_3d(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        z_coords = [p[2] for p in points]
        try:
            centroid_x = sum(x_coords) / len(points)
            centroid_y = sum(y_coords) / len(points)
            centroid_z = sum(z_coords) / len(points)
        except:
            print("ZeroDivisionError: division by zero")
        return (centroid_x, centroid_y, centroid_z)

    # 点集合质心坐标计算
    def calculate_centroid(self, points):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        try:
            centroid_x = sum(x_coords) / len(points)
            centroid_y = sum(y_coords) / len(points)
            return (centroid_x, centroid_y)
        except ZeroDivisionError:
            return ()