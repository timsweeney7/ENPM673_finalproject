import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
import json


def data_set_setup(sequence) -> tuple:
    """
    return:  (left_images_list, right_images_list, P0, P1, groundTruth, times)
    """

    seq_dir = f'./kittiDataSet/sequences/{sequence}/'
    poses_dir = f'./kittiDataSet/poses/{sequence}.txt'
    poses = pd.read_csv(poses_dir, delimiter=' ', header=None)

    # Get names of files to iterate through
    left_image_files = os.listdir(seq_dir + 'image_0')
    left_image_files.sort()
    right_image_files = os.listdir(seq_dir + 'image_1')
    right_image_files.sort()

     # Get calibration details for scene
    calib = pd.read_csv(seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
    P0 = np.array(calib.loc['P0:']).reshape((3,4)) # left 
    P1 = np.array(calib.loc['P1:']).reshape((3,4)) # right

    # Get times and ground truth poses
    times = np.array(pd.read_csv(seq_dir + 'times.txt', delimiter=' ', header=None))
    gt = np.zeros((len(poses), 3, 4))
    for i in range(len(poses)):
        gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

    # get first images --- Currently not used
    first_image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[0], cv2.IMREAD_UNCHANGED)
    first_image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[0], cv2.IMREAD_UNCHANGED)
    imheight = first_image_left.shape[0]
    imwidth = first_image_left.shape[1]

    return (left_image_files, right_image_files, P0, P1, gt, times)



def algorithm_1(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    #f = k_left[0][0]            # focal length of x axis for left camera
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]

    # Choose stereo matching algorithm
    matcher_name = 'sgbm'



    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        sad_window = 6
        num_disparities = sad_window * 16
        block_size = 11
            
        
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 1 * block_size ** 2,
                                        P2 = 32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            

        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        
        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        

        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)

        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top
            
        # Estimate motion between sequential images of the left camera
        
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time


""" --- Replacing StereoSGBM with StereoBM --- """
def algorithm_2(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]



    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        

        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top
            
        # Estimate motion between sequential images of the left camera
        
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time


""" --- Only taking the top 100 matches for motion estimation ---"""
def algorithm_3(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]



    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        

        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # Only take the top 100 matches
        if(len(matches)>100):
            matches = matches[:100]
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time


""" --- using Lowe's Ratio test to determine good matches --- """
""" ---         using StereoBM      ----  """
def algorithm_4(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        # matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.3*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time

""" --- using Lowe's Ratio test to determine good matches --- """
""" ---         using StereoSGBM      ----"""
def algorithm_5(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
         
            
        matcher = cv2.StereoSGBM_create(numDisparities=80,
                                        minDisparity=3,
                                        blockSize=21,
                                        P1 = 8 * 1 * 21 ** 2,
                                        P2 = 32 * 1 * 21 ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            

        disp = matcher.compute(image_left, image_right).astype(np.float32)/16            
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.SIFT_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        # matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.3*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time


""" --- Using ORB for feature detection instead of SIFT  --- """
""" ---         using StereoBM      ----  """
def algorithm_6(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        matcher = cv2.StereoBM_create()

        matcher.setNumDisparities(80)
        matcher.setBlockSize(21)
        matcher.setPreFilterCap(11)
        matcher.setUniquenessRatio(0)
        matcher.setSpeckleRange(1)
        matcher.setSpeckleWindowSize(0)
        matcher.setDisp12MaxDiff(14)
        matcher.setMinDisparity(3)
        matcher.setPreFilterType(0)
        matcher.setPreFilterSize(17)
        matcher.setTextureThreshold(0)   
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.ORB_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        #matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.65*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time

""" --- Using ORB for feature detection instead of SIFT  --- """
""" ---         using StereoSGBM      ----  """
def algorithm_7(start_pose:int = None, end_pose:int = None):
    
    if(end_pose == None):
        end_pose = len(left_image_files)
    if(start_pose == None):
        start_pose = 0
    num_frames = end_pose - start_pose

    # statistics for algo execution
    total_time = 0

    # Decompose left/right camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    # Get constant values for algorithm 
    b = t_right[0] - t_left[0]  #  baseline of stereo pair
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]


    # Establish homogeneous transformation matrix. First pose is ground truth    
    T_tot = gt[start_pose]
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]


    for i in range(num_frames - 1):
        # Stop if we've reached the second to last frame, since we need two sequential frames

        # Start timer for frame
        start = time.time()
        # Get our stereo images for depth estimation
        seq_dir = f'./kittiDataSet/sequences/{sequence}/'
        image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[start_pose + i], cv2.IMREAD_UNCHANGED)
        image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[start_pose+ i +1], cv2.IMREAD_UNCHANGED)  
        
        
        matcher = cv2.StereoSGBM_create(numDisparities=80,
                                        minDisparity=3,
                                        blockSize=21,
                                        P1 = 8 * 1 * 21 ** 2,
                                        P2 = 32 * 1 * 21 ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            
        disp = matcher.compute(image_left, image_right).astype(np.float32)/16       
        

        # Avoid instability and division by zero
        disp[disp == 0.0] = 0.1
        disp[disp == -1.0] = 0.1
        
        # Make empty depth map then fill with depth
        depth = np.ones(disp.shape)
        depth = fx * b / disp
        
        # Get keypoints and descriptors for left camera image of two sequential frames
        det = cv2.ORB_create()
        kp0, des0 = det.detectAndCompute(image_left,None)
        kp1, des1 = det.detectAndCompute(image_plus1,None)
        
        # Get matches between features detected in the two images
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(des0, des1, k=2)
        #matches = sorted(matches, key = lambda x:x[0].distance) # sort the matches with lowest distance at top

        # ratio test as per Lowe's paper.  Remove points that fail the ratio test
        delete = []
        for ii,(m,n) in enumerate(matches):
            if m.distance > 0.65*n.distance:
                delete.append(ii)
        matches = np.delete(matches, delete, 0)
            
        # Estimate motion between sequential images of the left camera
        rmat = np.eye(3)
        tvec = np.zeros((3, 1))
        
        image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
        image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
        
        object_points = np.zeros((0, 3))
        delete = []

        # Extract depth information of query image at match points and build 3D positions
        for j, (u, v) in enumerate(image1_points):
            z = depth[int(v), int(u)]
            # prune points with a depth greater than a specified limit because they are erroneous
            if z > 3000:
                delete.append(j)
                continue
                
            # Use arithmetic to extract x and y (faster than using inverse of k)
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            #object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)
        
        # Use PnP algorithm with RANSAC to compute image 2 transformation from image 1
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
        
        # Convert from Rodriques format to rotation matrix format
        rmat = cv2.Rodrigues(rvec)[0]
        
        # Create blank homogeneous transformation matrix
        Tmat = np.eye(4)
        # Place resulting rotation matrix  and translation vector in their proper locations
        # in homogeneous T matrix
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot @ np.linalg.inv(Tmat)
            
        # Place pose estimate in i+1 to correspond to the second image, which we estimated for
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        # End the timer for the frame and report frame rate to user
        end = time.time()
        computation_time = end-start
        total_time += computation_time
        mean_time = total_time/(i+1)

        print(f'Time to compute frame {i+1}: {np.round(end-start, 3)}s      Mean time: {mean_time}') 
        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='r')
        plt.pause(1e-32)

    # end of algorithm, return results
    print(f"Program execution time: {total_time}s")
    plt.plot(xs, ys, zs, c='r')
    plt.waitforbuttonpress()
    return trajectory, mean_time, total_time

def save_results(results, gt, mean_time, total_time, path):

    results_writable = []
    for i in range(len(results)):
        results_writable.append(list(np.ravel(results[i])))

    gt_writable = []
    for i in range(len(gt)):
        gt_writable.append(list(np.ravel(gt[i]))) 
    
    data_to_write = {
        "odometry" : results_writable,
        "ground truth": gt_writable,
        "mean time" : mean_time,
        "total time" : total_time
    }

    with open(path, "w") as outfile:
        json.dump(data_to_write, outfile)

def compute_error(gt,computed_trajectory,start_pose):
    gt = gt[start_pose:,:,3]
    computed_trajectory = computed_trajectory[:,:,3]
    abserror = []
    relerror = []
    angerror = []
    for i in range(len(computed_trajectory)):
        abserror.append(abs(np.linalg.norm(gt[i]-computed_trajectory[i])))
    for j in range(1,len(computed_trajectory)-1):
        relerror.append(abs(np.linalg.norm(abs(np.linalg.norm(gt[j]-gt[j+1]))-abs(np.linalg.norm(computed_trajectory[j]-computed_trajectory[j+1])))))
        ba1 = gt[j-1] - gt[j]
        bc1 = gt[j+1] - gt[j]
        cosine_angle1 = np.dot(ba1, bc1) / (np.linalg.norm(ba1) * np.linalg.norm(bc1))
        angle1 = np.arccos(cosine_angle1)

        ba2 = computed_trajectory[j-1] - computed_trajectory[j]
        bc2 = computed_trajectory[j+1] - computed_trajectory[j]
        cosine_angle2 = np.dot(ba2, bc2) / (np.linalg.norm(ba2) * np.linalg.norm(bc2))
        angle2 = np.arccos(cosine_angle2)

        angerror.append(abs(np.degrees(angle1)-np.degrees(angle2)))
    return abserror,relerror,angerror

if __name__ == "__main__":
    
    start_time = datetime.now()
    sequence = "00"
    left_image_files, right_image_files, P0, P1, gt, times = data_set_setup(sequence)
    start_pose = 950
    end_pose  = 1000
    # Setup plot that will be used on each iteration of code
    fig = plt.figure(figsize=(14, 14))
    plt.title("Trajectory")
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = gt[:, 0, 3]
    ys = gt[:, 1, 3]
    zs = gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='b')
    
    # Run algorithm
    computed_trajectory, mean_time, total_time = algorithm_7(start_pose, end_pose)

    # Compute Error 
    abserror,relerror,angerror = compute_error(gt, computed_trajectory,start_pose)
    
    fig2 = plt.figure(figsize=(10,10))
    ax2 = fig2.add_subplot()
    plt.title("Absolute Error")
    ax2.plot(range(len(abserror)),abserror)
    plt.xlabel("Frame")
    plt.ylabel("Absolute Error (meters)")
    plt.waitforbuttonpress()

    fig3 = plt.figure(figsize=(10,10))
    ax3 = fig3.add_subplot()
    plt.title("Relative Error")
    ax3.plot(range(len(relerror)),relerror)
    plt.xlabel("Frame")
    plt.ylabel("Relative Error (meters)")
    plt.waitforbuttonpress()

    fig4 = plt.figure(figsize=(10,10))
    ax4 = fig4.add_subplot()
    plt.title("Relative Heading Angle Error")
    ax4.plot(range(len(angerror)),angerror)
    plt.xlabel("Frame")
    plt.ylabel("Relative Heading Angle Error (degrees)")
    plt.waitforbuttonpress()

    # Save results
    path = "./kittiDataSet/results/algorithm_1.json"
    save_results(computed_trajectory, gt, mean_time, total_time, path)
    

    
