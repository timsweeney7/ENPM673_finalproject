import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import glob
import ast
from datetime import datetime
import json


def calibrate_camera(path):
    CHESSBOARD = (9,6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each chessrboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each chessboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    # Extracting path of individual image stored in a given directory
    images = glob.glob(path)
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv.findChessboardCorners(gray, CHESSBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of chess board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(image= gray, corners= corners, winSize=(11,11), zeroZone= (-1,-1), criteria= criteria)
            imgpoints.append(corners2)

    """
    Performing camera calibration by passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the detected corners (imgpoints)
    """
    leftRet, mtx, dist, leftRvecs, leftTvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h,w = img.shape[:2]
    newcameramtx, leftroi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    return mtx, dist, newcameramtx
   
def calculate_error():
    print()

def extractParams(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    camMatrix = data["cameraMatrix"]
    camMatrix = np.reshape(camMatrix, (3,3))

    distortionCoef = data["distortionCoef"]
    distortionCoef = np.array(distortionCoef)

    newCamMatrix = data["newCameraMatrix"]
    newCamMatrix = np.array(newCamMatrix)

    return camMatrix, distortionCoef, newCamMatrix

def drawlines(img, lines, pts):
    """
    Given a list of points and lines, draw them on an image
    inputs:
    img - the image to draw on
    lines - list of lines in (a,b,c) format corresponding to ax + by + c = 0
    pts - list of points, where each point is associated with line of same index
    outputs:
    img - the image that was drawn on
    """
    r, c = img.shape
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    for line, pt in zip(lines, pts):

        # get a random color
        color = tuple(np.random.randint(0, 255, 3).tolist())


        # select points on the left and right side of the image
        # calculate associated y value
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1] ])

        # Color the points
        img = cv.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv.circle(img, tuple(pt), 5, color, -1)

    return img

def detectMatches_bruteforce(img1,img2):
    """
    Detect matches between pictures using feature matching
    inputs:
    img1 - image 1
    img2 - image 2
    outputs:
    imageMatches - the matching points on the combined image
    pts1 - points from img1
    pts2 - points from img2
    """
    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1,None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2,None)

    # Initialize the feature matcher using brute-force matching
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the top 50 matches
    numMatches = 40
    imageMatches = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:numMatches], None)
    pts1 = np.float32([keypoints1[match.queryIdx].pt for match in matches])
    pts2 = np.float32([keypoints2[match.trainIdx].pt for match in matches])

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return imageMatches,pts1,pts2


def detectMatches_flann(img1, img2, k:int = None):
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    pts1 = []
    pts2 = []
    good = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.4*n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            good.append([m])

    matches_mask = None
    if k is not None:
        matches_mask = [[0]] * len(good)
        matches_mask[0:k] = [[1]] * k

    # drawing nearest neighbours
    imgMatches = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, matchesMask=matches_mask)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    return imgMatches, pts1, pts2


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{datetime.now() - start_time}] Starting image pipeline")

    """ --- Uncomment to recalibrate cameras --- """
    # left_mtx, left_dst_coef, left_newcameramtx = calibrate_camera(path = "./images/calibration/leftCal")
    # right_mtx, right_dst_coef, right_newcameramtx = calibrate_camera(path = "./images/calibration/rightCal")

    """ --- Uncomment to load calibration from file --- """
    left_mtx, left_dst_coef, left_newcameramtx  = extractParams("./calibrationData/leftCal.json")
    right_mtx, right_dst_coef, right_newcameramtx  = extractParams("./calibrationData/rightCal.json")


    imgLeft = cv.imread("./images/flight/leftFlight/flightleft_out_0000000510.png")
    imgRight = cv.imread("./images/flight/rightFlight/flightright_out_0000000510.png")

    #matchesImg1, pts1, pts2 = detectMatches_bruteforce(img1= imgLeft, img2= imgRight)
    matchesImg2, pts1, pts2 = detectMatches_flann(img1= imgLeft, img2= imgRight, k=10)
    cv.imshow("matches",matchesImg2)

    #get fundamental matrix
    # changing from LMEDS to RANSAC produces wildly different results
    F, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
    #get essential matrix
    E = right_mtx.T @ F @ left_mtx
    print(E)
    #_,r,t,_ = cv.recoverPose(E,pts1,pts2,cameraMatrix=left_mtx)
    # using this function because we have two different camera matrices
    numberOfInliers, E, R, t, mask = cv.recoverPose(pts1, pts2, left_mtx, left_dst_coef, right_mtx, right_dst_coef, method=cv.LMEDS)
    print(E)
    # we see here that the two different methods return different E matrix????
    # something is wrong at this point

    #only keep inlier points from the fundamental matrix calculation
    #ptsLeft = pts1[inliers.ravel()==1]
    #ptsRight = pts2[inliers.ravel()==1]

    h1, w1, _ = imgLeft.shape
    h2, w2, _ = imgRight.shape

    #rectify the images by getting R and P matrices to be used to undistort in next step
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(left_mtx, np.zeros((1,5)),right_mtx, np.zeros((1,5)),[w1,h1], r, t, alpha=1)

    # Generate rectification maps
    map1_left, map2_left = cv.initUndistortRectifyMap(left_mtx, np.zeros((1,5)), R1, P1[:,:-1], [w1,h1], cv.CV_32FC1)
    map1_right, map2_right = cv.initUndistortRectifyMap(right_mtx, np.zeros((1,5)), R2, P2[:,:-1], [w2,h2], cv.CV_32FC1)

    # Rectify left and right images
    img1_rect = cv.remap(imgLeft, map1_left, map2_left, cv.INTER_LINEAR)
    img2_rect = cv.remap(imgRight, map1_right, map2_right, cv.INTER_LINEAR)


    cv.imshow("im1", img1_rect)
    cv.imshow("im2", img2_rect)

    print(f"[{datetime.now() - start_time}] Pipeline complete!")
    cv.waitKey()