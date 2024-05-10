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



def algorith_1(start_pose:int = None, end_pose:int = None):
    
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
            #x = z*(u-cx)/fx
            #y = z*(v-cy)/fy
            #object_points = np.vstack([object_points, np.array([x, y, z])])
            # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
            #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])
            object_points = np.vstack([object_points, np.linalg.inv(k_left) @ (z * np.array([u, v, 1]))])

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
    return trajectory


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


if __name__ == "__main__":
    
    start_time = datetime.now()
    sequence = "00"
    left_image_files, right_image_files, P0, P1, gt, times = data_set_setup(sequence)

    """
    # Setup plot that will be used on each iteration of code
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = gt[:, 0, 3]
    ys = gt[:, 1, 3]
    zs = gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='b')
    
    # Run algorithm
    computed_trajectory = algorith_1(start_pose= 950, end_pose=1500)

    # Compute Error 
    #  error = compute_error(gt, computed_trajectory)

    # Save results
    # save_results(computed_trajectory, error, path_for_save_file)
    """
    fake_data = np.zeros((1,3,4))
    fake_data_list = np.zeros((0,3,4))
    for i in range(3):
        fake_data[0,2,3] = i
        fake_data_list = np.vstack([fake_data_list, fake_data])
    fake_data = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
    fake_data_list = np.vstack([fake_data_list, fake_data])
    path = './kittiDataSet/results/test_output.json'

    save_results(fake_data_list, gt, 0, 3000, path)

    
