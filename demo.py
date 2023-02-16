import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os 
import pandas as pd
import time
#%%

class dataset():
    def __init__(self,seq="00"):
        
        self.sequences_dir = 'sequences/{}/'.format(seq)
        self.poses_dir = 'poses/{}.txt'.format(seq)
        
        self.left_img=(cv2.imread(self.sequences_dir+"image_0/"+i,0) for i in (os.listdir(self.sequences_dir+"image_0")) if i.endswith("png"))
        self.right_img=(cv2.imread(self.sequences_dir+"image_1/"+i,0) for i in (os.listdir(self.sequences_dir+"image_1")) if i.endswith("png"))
        self.poses=pd.read_csv(self.poses_dir, delimiter=' ', header=None)
        self.calib=pd.read_csv(self.sequences_dir+"calib.txt", delimiter=' ', header=None, index_col=0)
        
        self.p0=np.array(self.calib.iloc[0]).reshape((3,4))
        self.p1=np.array(self.calib.iloc[1]).reshape((3,4))
        self.Tr = np.array(self.calib.iloc[-1]).reshape((3,4))


    def next_imgs(self):
        return next(self.left_img),next(self.right_img)
    def reset_images(self):
        self.left_img=(cv2.imread(self.sequences_dir+"image_0/"+i,0) for i in (os.listdir(self.sequences_dir+"image_0")) if i.endswith("png"))
        self.right_img=(cv2.imread(self.sequences_dir+"image_1/"+i,0) for i in (os.listdir(self.sequences_dir+"image_1")) if i.endswith("png"))
    
data=dataset()



#%% Visualization left camera's frames
for i in range(len(os.listdir(data.sequences_dir+"image_0"))):
    cv2.imshow("winname", data.next_imgs()[0])
    key=cv2.waitKey(1)
    if key==ord('q'):
        data.reset_images()
        break
cv2.destroyAllWindows()

#%%
def compute_left_disparity_map(img_left, img_right, matcher='bm', verbose=False):
    
    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 1 * block_size ** 2,
                                        P2 = 32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    

        
    start = time.perf_counter()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = time.perf_counter()
    
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}', end-start)
        
    return disp_left

disp = compute_left_disparity_map(data.next_imgs()[0],
                                  data.next_imgs()[1],
                                  matcher='sgbm',
                                  verbose=True)
plt.figure(figsize=(11,7))
plt.imshow(disp);
#%% Visualization disp frames
for i in range(len(os.listdir(data.sequences_dir+"image_0"))):
    disp = compute_left_disparity_map(data.next_imgs()[0],
                                      data.next_imgs()[1],
                                      matcher='sgbm',
                                      verbose=True)
    disp/=disp.max()
    disp*=255.
    cv2.imshow("winname", disp.astype(np.uint8))
    key=cv2.waitKey(27)
    if key==ord('q'):
        data.reset_images()
        break
cv2.destroyAllWindows()
#%%
def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    return k, r, t

def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    
    if rectified:
        b = t_right[0] - t_left[0]
    else:
        b = t_left[0] - t_right[0]
        
    f = k_left[0][0]
    
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left
    
    return depth_map

k_left, r_left, t_left = decompose_projection_matrix(data.p0)
k_right, r_right, t_right = decompose_projection_matrix(data.p1)

depth = calc_depth_map(disp, k_left, t_left, t_right)
#%%
for i in range(len(os.listdir(data.sequences_dir+"image_0"))):
    k_left, r_left, t_left = decompose_projection_matrix(data.p0)
    k_right, r_right, t_right = decompose_projection_matrix(data.p1)
    disp = compute_left_disparity_map(data.next_imgs()[0],
                                      data.next_imgs()[1],
                                      matcher='sgbm',
                                      verbose=True)
    depth = calc_depth_map(disp, k_left, t_left, t_right)
    depth/=depth.max()
    depth*=255
    cv2.imshow("winname", disp.astype(np.uint8))
    key=cv2.waitKey(27)
    if key==ord('q'):
        data.reset_images()
        break
cv2.destroyAllWindows()
#%%
