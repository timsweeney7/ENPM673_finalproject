import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class Dataset_Handler():
    def __init__(self, sequence, progress_bar=True, low_memory=True):
        
        # This will tell odometry functin how to access data from this object
        self.low_memory = low_memory
        
        # Set file paths and get ground truth poses
        self.seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        self.poses_dir = f'./kittiDataSet/poses/{sequence}.txt'
        poses = pd.read_csv(self.poses_dir, delimiter=' ', header=None)
        
        # Get names of files to iterate through
        self.left_image_files = os.listdir(self.seq_dir + 'image_0')
        self.left_image_files.sort()
        self.right_image_files = os.listdir(self.seq_dir + 'image_1')
        self.right_image_files.sort()
        self.num_frames = len(self.left_image_files)
        
        # Get calibration details for scene
        # P0 and P1 are Grayscale cams, P2 and P3 are RGB cams
        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
        self.P0 = np.array(calib.loc['P0:']).reshape((3,4)) # left 
        self.P1 = np.array(calib.loc['P1:']).reshape((3,4)) # right
        
        # Get times and ground truth poses
        self.times = np.array(pd.read_csv(self.seq_dir + 'times.txt', 
                                          delimiter=' ', 
                                          header=None))
        self.gt = np.zeros((len(poses), 3, 4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3, 4))
        
        # Get images and lidar loaded
        if self.low_memory:
            # Will use generators to provide data sequentially to save RAM
            # Use class method to set up generators
            self.reset_frames()
            # Store original frame to memory for testing functions
            self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[0], 0)
            self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' + self.right_image_files[0], 0)
            self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1], 0)
            self.second_image_right = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1], 0)
            
            self.imheight = self.first_image_left.shape[0]
            self.imwidth = self.first_image_left.shape[1]
            
            
    def reset_frames(self):
        # Resets all generators to the first frame of the sequence
        self.images_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left, 0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right, 0)
                            for name_right in self.right_image_files)
        pass


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
        
    start = datetime.datetime.now()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = datetime.datetime.now()
    
    if verbose:
        print(f'Time to compute disparity map using Stereo{matcher_name.upper()}', end-start)
        
    return disp_left


if __name__ == "__main__":
    handler = Dataset_Handler('00')

    cv2.imshow("image", handler.first_image_left)

    h,w = handler.first_image_left.shape
    disp = compute_left_disparity_map(handler.first_image_left,
                                  handler.first_image_right,
                                  matcher='bm',
                                  verbose=True)
    #plt.figure(figsize=(11,7))
    cv2.imshow("disparity", disp)

    cv2.waitKey()