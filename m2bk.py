import os

import math
import numpy as np
import cv2 as cv

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
            
        
def visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=False):
    image1 = image1.copy()
    image2 = image2.copy()
    
    for i in range(0, len(image1_points)):
        # Coordinates of a point on t frame
        p1 = (int(image1_points[i][0]), int(image1_points[i][1]))
        # Coordinates of the same point on t+1 frame
        p2 = (int(image2_points[i][0]), int(image2_points[i][1]))

        cv.circle(image1, p1, 5, (0, 255, 0), 1)
        cv.arrowedLine(image1, p1, p2, (0, 255, 0), 1)
        cv.circle(image1, p2, 5, (255, 0, 0), 1)

        if is_show_img_after_move:
            cv.circle(image2, p2, 5, (255, 0, 0), 1)
    
    if is_show_img_after_move: 
        return image2
    else:
        return image1
