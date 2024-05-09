import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from datetime import datetime

start_time = datetime.now()
sequence = "00"
seq_dir = f'./kittiDataSet/sequences/{sequence}/'
poses_dir = f'./kittiDataSet/poses/{sequence}.txt'
poses = pd.read_csv(poses_dir, delimiter=' ', header=None)

# Get names of files to iterate through
left_image_files = os.listdir(seq_dir + 'image_0')
left_image_files.sort()
right_image_files = os.listdir(seq_dir + 'image_1')
right_image_files.sort()
num_frames = len(left_image_files)


 # Get calibration details for scene
calib = pd.read_csv(seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)
P0 = np.array(calib.loc['P0:']).reshape((3,4)) # left 
P1 = np.array(calib.loc['P1:']).reshape((3,4)) # right

# Get times and ground truth poses
times = np.array(pd.read_csv(seq_dir + 'times.txt', delimiter=' ', header=None))
gt = np.zeros((len(poses), 3, 4))
for i in range(len(poses)):
    gt[i] = np.array(poses.iloc[i]).reshape((3, 4))

first_image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[0], cv2.IMREAD_UNCHANGED)
first_image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[0], cv2.IMREAD_UNCHANGED)

imheight = first_image_left.shape[0]
imwidth = first_image_left.shape[1]
 # Establish homogeneous transformation matrix. First pose is identity    
T_tot = np.eye(4)
trajectory = np.zeros((num_frames, 3, 4))
trajectory[0] = T_tot[:3, :]

# Decompose left camera projection matrix to get intrinsic k matrix
k_left, r_left, t_left,_,_,_,_ = cv2.decomposeProjectionMatrix(P0)
t_left = (t_left / t_left[3])[:3]

for i in range(num_frames - 1):
    # Stop if we've reached the second to last frame, since we need two sequential frames

    # Start timer for frame
    start = datetime.now()
    # Get our stereo images for depth estimation
    image_left = cv2.imread(seq_dir + 'image_0/' + left_image_files[i], cv2.IMREAD_UNCHANGED)
    image_right = cv2.imread(seq_dir + 'image_1/' + right_image_files[i], cv2.IMREAD_UNCHANGED)
    image_plus1 = cv2.imread(seq_dir + 'image_0/' + left_image_files[i+1], cv2.IMREAD_UNCHANGED)    # Estimate depth if using stereo depth estimation
    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = 'sgbm'
    
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

    # Decompose projection matrices
    k_left, r_left, t_left, _, _, _, _ = cv2.decomposeProjectionMatrix(P0)
    t_left = (t_left / t_left[3])[:3]
    k_right, r_right, t_right, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    t_right = (t_right / t_right[3])[:3]
    
    # Calculate depth map for left camera
    # Get focal length of x axis for left camera
    f = k_left[0][0]
    
    # Calculate baseline of stereo pair
    b = t_right[0] - t_left[0] 
        
    # Avoid instability and division by zero
    disp[disp == 0.0] = 0.1
    disp[disp == -1.0] = 0.1
    
    # Make empty depth map then fill with depth
    depth = np.ones(disp.shape)
    depth = f * b / disp
    



    # Get keypoints and descriptors for left camera image of two sequential frames
    det = cv2.SIFT_create()
    kp0, des0 = det.detectAndCompute(image_left,None)
    kp1, des1 = det.detectAndCompute(image_plus1,None)

    
    # Get matches between features detected in the two images
    matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(des0, des1, k=2)
    matches = sorted(matches, key = lambda x:x[0].distance)
        
    # Estimate motion between sequential images of the left camera
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp0[m.queryIdx].pt for (m,n) in matches])
    image2_points = np.float32([kp1[m.trainIdx].pt for (m,n) in matches])
    
    cx = k_left[0, 2]
    cy = k_left[1, 2]
    fx = k_left[0, 0]
    fy = k_left[1, 1]
    object_points = np.zeros((0, 3))
    delete = []

    # Extract depth information of query image at match points and build 3D positions
    for i, (u, v) in enumerate(image1_points):
        z = depth[int(v), int(u)]
        # If the depth at the position of our matched feature is above 3000, then we
        # ignore this feature because we don't actually know the depth and it will throw
        # our calculations off. We add its index to a list of coordinates to delete from our
        # keypoint lists, and continue the loop. After the loop, we remove these indices
        if z > 3000:
            delete.append(i)
            continue
            
        # Use arithmetic to extract x and y (faster than using inverse of k)
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy
        object_points = np.vstack([object_points, np.array([x, y, z])])
        # Equivalent math with dot product w/ inverse of k matrix, but SLOWER (see Appendix A)
        #object_points = np.vstack([object_points, np.linalg.inv(k).dot(z*np.array([u, v, 1]))])

    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    
    # Use PnP algorithm with RANSAC for robustness to outliers
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k_left, None)
    #print('Number of inliers: {}/{} matched features'.format(len(inliers), len(match)))
    
    # Above function returns axis angle rotation representation rvec, use Rodrigues formula
    # to convert this to our desired format of a 3x3 rotation matrix
    rmat = cv2.Rodrigues(rvec)[0]
    


    # Create blank homogeneous transformation matrix
    Tmat = np.eye(4)
    # Place resulting rotation matrix  and translation vector in their proper locations
    # in homogeneous T matrix
    Tmat[:3, :3] = rmat
    Tmat[:3, 3] = tvec.T
    
    T_tot = T_tot.dot(np.linalg.inv(Tmat))
        
    # Place pose estimate in i+1 to correspond to the second image, which we estimated for
    trajectory[i+1, :, :] = T_tot[:3, :]
    # End the timer for the frame and report frame rate to user
    end = datetime.now()
    print('Time to compute frame {}:'.format(i+1), end-start)
    
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = gt[:, 0, 3]
    ys = gt[:, 1, 3]
    zs = gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='k')



    xs = trajectory[:i+2, 0, 3]
    ys = trajectory[:i+2, 1, 3]
    zs = trajectory[:i+2, 2, 3]
    plt.plot(xs, ys, zs, c='chartreuse')
    plt.pause(1e-32)
    plt.waitforbuttonpress()
