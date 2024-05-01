import cv2
import math
import numpy as np

class mathProc:
    def __init__(self) -> None:
        pass

    def calculate_distance(point1, point2):
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)