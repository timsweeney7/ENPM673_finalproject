import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import glob
import ast


def func1():
    CHESSBOARD = (9,6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each chessrboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each chessboard image
    imgpoints = []


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    leftImages = glob.glob('./images/calibration/leftCal/*.png')
    for fname in leftImages:
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
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    leftRet, leftMtx, leftDist, leftRvecs, leftTvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:")
    print(leftMtx)
    print("dist:")
    print(leftDist)

    h,w = img.shape[:2]
    leftnewcameramtx, leftroi = cv.getOptimalNewCameraMatrix(leftMtx, leftDist, (w,h), 1, (w,h))
    print("New Camera Matrix:")
    print(leftnewcameramtx)

    dst = cv.undistort(img, leftMtx, leftDist, None, leftnewcameramtx)
    cv.imshow("Left Image", dst)

   



if __name__ == "__main__":
    print("Starting image pipeline")
    func1()
    cv.waitKey()