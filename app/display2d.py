import utils
from utils import Jerk
from data import dataset
import cv2
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import sys

class Display2d:
    def __init__(self):
        # create figure
        fig = plt.figure(figsize=(8, 4))
        fig.canvas.manager.window.wm_geometry("+0+400")
        # set ax1 and ax1 variables
        self.ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        self.ax1.set_title('Road Estimation')
        self.ax1.view_init(elev=-20, azim=270)
        xs = data.gt[:, 0, 3]
        ys = data.gt[:, 1, 3]
        zs = data.gt[:, 2, 3]
        self.ax1.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        self.ax1.plot(xs, ys, zs, c='k', label='Ground truth')
        self.ax1.plot(0, 0, 0, c='chartreuse', label='Estimated')
        ax1.legend(fontsize=8, )

        # set ax2 and ax2 variables
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Z axis')
        ax2.set_title('Jerk Estimation')
        self.text_est = ax2.text(-35, 40, '',color='b', ha='left')
        self.text_gt = ax2.text(-35, 34, '',color='g', ha='left')
        ax2.set_xlim(-35, 35)
        ax2.set_ylim(-35, 45)
        ax2.set_aspect('equal')
        ellipse_static = Ellipse(
            xy=(0, 0), width=30, height=60, angle=0, alpha=0.5, fill=False, edgecolor='r')
        ax2.add_artist(ellipse_static)
        self.rect = Rectangle((-2, -2), 4, 4, linewidth=1,
                         edgecolor='b', fill=False)
        self.rect_gt = Rectangle((-2, -2), 4, 4, linewidth=1,
                    edgecolor='g', fill=False)
        ax2.add_artist(self.rect)
        ax2.add_artist(self.rect_gt)

        fig.subplots_adjust(wspace=0.5)

    def UpdataPlot(self,jerk_gt,jerk_est,xs,ys,zs):
        self.ax1.plot(xs, ys, zs, c='chartreuse', label='estimated')
        self.rect_gt.set_xy((jerk_gt[0] - 2, jerk_gt[2] - 2))
        self.text_gt.set_text(
            f"jerk_gt_mag= {np.linalg.norm([jerk_gt[0], jerk_gt[2]]):0.2f}$ m/s^3$")
        self.rect.set_xy((jerk_est[0] - 2, jerk_est[2] - 2))
        self.text_est.set_text(
            f"jerk_est_mag= {np.linalg.norm([jerk_est[0], jerk_est[2]]):0.2f}$ m/s^3$")
        plt.pause(1e-32)