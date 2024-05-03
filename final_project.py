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
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    leftImages = glob.glob(path)
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
    print(type(camMatrix))

    distortionCoef = data["distortionCoef"]
    distortionCoef = np.array(distortionCoef)

    return camMatrix, distortionCoef


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




if __name__ == "__main__":
    start_time = datetime.now()
    print(f"[{datetime.now() - start_time}] Starting image pipeline")

    left_mtx, left_dst_coef  = extractParams("./calibrationData/leftCal.json")
    right_mtx, right_dst_coef  = extractParams("./calibrationData/rightCal.json")

    


    print(f"[{datetime.now() - start_time}] Pipeline complete!")
    cv.waitKey()