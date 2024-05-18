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
            return (centroid_x, centroid_y, centroid_z)
        except:
            print("ZeroDivisionError: division by zero")

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

    def getCircleCenter(self, box):
        x, y = 0
        for point in box:
            x = x + point[0]
            y = y + point[1]
        # 点坐标， 半径
        return [(x/2, y/2), x/2-x]

    def calcDotCenter(self, xyxy_list):
        try:
            x_coords = [xyxy_list[0], xyxy_list[2]]
            y_coords = [xyxy_list[1], xyxy_list[3]]
        except IndexError:
            return ()
        try:
            center_x = int(sum(x_coords) / 2)
            center_y = int(sum(y_coords) / 2)
            return (center_x, center_y)
        except ZeroDivisionError:
            return ()