import utils
from data import dataset
import cv2
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle


def VisualOdometry(data, stereo_matcher='sgbm', detector='orb', matching='BF', GoodP=True, dist_threshold=0.5, subset=None, plot=False):
    if subset is not None:
        num_frame = subset
    else:
        num_frame = len(os.listdir(data.sequences_dir+"image_0"))
    if plot:
        # create figure
        fig = plt.figure(figsize=(15, 7))
        # set ax1 and ax1 variables
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title('Road Estimation')
        ax1.view_init(elev=-20, azim=270)
        xs = data.gt[:, 0, 3]
        ys = data.gt[:, 1, 3]
        zs = data.gt[:, 2, 3]
        ax1.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax1.plot(xs, ys, zs, c='k')
        
        # set ax2 and ax2 variables
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title('Jerk Estimation')
        text_obj = ax2.text(-3, 18, '', ha='right')
        ax2.set_xlim(-20, 20)
        ax2.set_ylim(-20, 20)
        ax2.set_aspect('equal')
        ellipse_static = Ellipse(
            xy=(0, 0), width=15, height=30, angle=0, alpha=0.5, fill=False, edgecolor='r')
        ax2.add_artist(ellipse_static)
        rect = Rectangle((-2, -2), 4, 4, linewidth=1,
                         edgecolor='b', fill=False)
        ax2.add_artist(rect)
    # get images from dataset
    img1_left, img1_right = data.next_imgs()

    # Decomposition the projection matrix 
    k_left, r_left, t_left = utils.decompose_projection_matrix(data.p0)
    k_right, r_right, t_right = utils.decompose_projection_matrix(data.p1)
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frame, 3, 4))
    trajectory[0] = T_tot[:3, :]
    total_time = 0
    position = np.zeros((3,))
    
    for idx in range(num_frame-1):
        print('----------------------------------------------------')
        start = time.perf_counter()
        img2_left, img2_right = data.next_imgs()
        # Calculate disparity map
        disp = utils.compute_left_disparity_map(
            img_left=img1_left, img_right=img1_right, matcher=stereo_matcher,)
        depht_map = utils.calc_depth_map(
            disp_left=disp, k_left=k_left, t_left=t_left, t_right=t_right)
        # Extract features
        kp1, des1 = utils.extract_features(
            img1_left, detector=detector, GoodP=GoodP,)
        kp2, des2 = utils.extract_features(
            img2_left, detector=detector, GoodP=GoodP,)
        # extract matches
        matches_unfilter = utils.match_features(
            des1=des1, des2=des2, matching=matching, detector=detector,)
        # filtering the matches
        if dist_threshold is not None:
            matches = utils.filter_matches_distance(
                matches_unfilter, dist_threshold=dist_threshold)
        else:
            matches = matches_unfilter
        # And estimate mation 
        rmat, tvec, image1_points, image2_points = utils.estimate_motion(
            matches=matches, kp1=kp1, kp2=kp2, k=k_left, depth1=depht_map)

        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        trajectory[idx+1, :, :] = T_tot[:3, :]
        end = time.perf_counter()
        total_time += (end-start)
        print('Time to compute frame {}:'.format(idx+1), end-start)
        position += T_tot[:-1, 3]
        jerkX, jerkY, _ = utils.estimate_jerk(
            position=position, time_diff=end-start)
        img1_left, img1_right = img2_left, img2_right
        if plot:
            # updete trajectory variables
            xs = trajectory[:idx+2, 0, 3]
            ys = trajectory[:idx+2, 1, 3]
            zs = trajectory[:idx+2, 2, 3]
            ax1.plot(xs, ys, zs, c='chartreuse')

            # updete ax2 variables
            rect.set_xy((jerkX-2, jerkY-2))
            text_obj.set_text(
                f"Jerk_mag= {np.linalg.norm([jerkX, jerkY]):0.4f}$*10e-8 ms^3$")
            plt.pause(1e-32)
            # show current left image
            cv2.imshow("current_left", img2_left)
            k = cv2.waitKey(30)
            if ord('q') == k:
                cv2.destroyAllWindows()
                break

    if plot:
        plt.close()

    avr_fps = num_frame/total_time
    return trajectory, avr_fps


# if __name__=="__main__":
data = dataset(seq="05")
trajectory, avr_fps = VisualOdometry(data=data, stereo_matcher='sgbm', detector='orb',
                                     matching='BF', GoodP=True, dist_threshold=0.5, subset=400, plot=True)

utils.visualize_trajectory(trajectory)

print("mse:", utils.mse(data.gt, trajectory))
print("average fps:", avr_fps)
cv2.destroyAllWindows()
