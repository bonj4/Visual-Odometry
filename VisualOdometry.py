import utils
from data import dataset
import cv2
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt


def VisualOdometry(data, stereo_matcher='sgbm', detector='orb', matching='BF', GoodP=True, dist_threshold=0.5, subset=None, plot=False):
    if subset is not None:
        num_frame = subset
    else:
        num_frame = len(os.listdir(data.sequences_dir+"image_0"))
    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = data.gt[:, 0, 3]
        ys = data.gt[:, 1, 3]
        zs = data.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')
    img1_left, img1_right = data.next_imgs()

    k_left, r_left, t_left = utils.decompose_projection_matrix(data.p0)
    k_right, r_right, t_right = utils.decompose_projection_matrix(data.p1)
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frame, 3, 4))
    trajectory[0] = T_tot[:3, :]
    total_time = 0
    for idx in range(num_frame-1):
        start = time.perf_counter()

        img2_left, img2_right = data.next_imgs()
        disp = utils.compute_left_disparity_map(
            img_left=img1_left, img_right=img1_right, matcher=stereo_matcher,)
        depht_map = utils.calc_depth_map(
            disp_left=disp, k_left=k_left, t_left=t_left, t_right=t_right)
        kp1, des1 = utils.extract_features(
            img1_left, detector=detector, GoodP=GoodP,)
        kp2, des2 = utils.extract_features(
            img2_left, detector=detector, GoodP=GoodP,)
        matches_unfilter = utils.match_features(
            des1=des1, des2=des2, matching=matching, detector=detector,)

        if dist_threshold is not None:
            matches = utils.filter_matches_distance(
                matches_unfilter, dist_threshold=dist_threshold)
        else:
            matches = matches_unfilter

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

        img1_left, img1_right = img2_left, img2_right
        if plot:
            cv2.imshow("winname", img2_left)
            cv2.waitKey(1)
            xs = trajectory[:idx+2, 0, 3]
            ys = trajectory[:idx+2, 1, 3]
            zs = trajectory[:idx+2, 2, 3]
            plt.plot(xs, ys, zs, c='chartreuse')
            plt.pause(1e-32)

    if plot:
        plt.close()

    avr_fps = num_frame/total_time
    return trajectory, avr_fps


if __name__=="__main__":
    data = dataset(seq="00")
    trajectory, avr_fps = VisualOdometry(data=data, stereo_matcher='bm', detector='orb',
                                        matching='BF', GoodP=True, dist_threshold=0.5, subset=100, plot=True)
    utils.visualize_trajectory(trajectory)
    print("mse:", utils.mse(data.gt, trajectory))
    print("average fps:", avr_fps)
    cv2.destroyAllWindows()
