import cv2 as cv
import numpy as np


def drawEpipolarlines(img1, lines, pts1):
    """
    Given a list of points, draw them and the corresponding epi-line on an image
    img1  - the image to draw on
    lines - list of lines in (a,b,c) format corresponding to ax + by + c = 0
    pts1  - list of points that index match lines index
    """
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)

    for line, pt1 in zip(lines, pts1):

        # get a random color
        color = tuple(np.random.randint(0, 255, 3).tolist())


        # select points on the left and right side of the image
        # calculate associated y value
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1] ])

        # Color the points
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)

    return img1



path = "./kittiDataSet/sequences/00"
image = "000950.png"

img1 = cv.imread(f'{path}/image_0/{image}', cv.IMREAD_GRAYSCALE) #queryimage # left image
#cv.namedWindow("img1", cv.WINDOW_NORMAL)
#cv.imshow("img1", img1)
img2 = cv.imread(f'{path}/image_0/{image}', cv.IMREAD_GRAYSCALE) #trainimage # right image
#cv.namedWindow("img2", cv.WINDOW_NORMAL)
#cv.imshow("img2",img2)


def nothing(x):
    pass

cv.namedWindow('disp', cv.WINDOW_NORMAL)
cv.resizeWindow('disp',600,1200)

# Creating an object of StereoBM algorithm
stereo = cv.StereoBM_create()
#stereo = cv.StereoSGBM.create()

cv.createTrackbar('minDisparity','disp',5,25,nothing)
cv.createTrackbar('numDisparities','disp',2,17,nothing)
cv.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv.createTrackbar('preFilterCap',5, 62, nothing)
cv.createTrackbar('blockSize','disp',5,50,nothing)
cv.createTrackbar('textureThreshold','disp',10,100,nothing)
cv.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv.createTrackbar('speckleRange','disp',0,100,nothing)
cv.createTrackbar('speckleWindowSize','disp',3,25,nothing)


if type(stereo) == cv.StereoBM:
    cv.createTrackbar('preFilterType','disp',1,1,nothing)
    cv.createTrackbar('preFilterSize','disp',2,25,nothing)

else:
    cv.createTrackbar('P1','disp', 0, 40, nothing)
    cv.createTrackbar('P2','disp', 0, 40, nothing)


 

while True:
    
    # Updating the parameters based on the trackbar positions
    numDisparities = cv.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterCap = cv.getTrackbarPos('preFilterCap','disp')
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv.getTrackbarPos('minDisparity','disp')

    if type(stereo) == cv.StereoBM:
        preFilterType = cv.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv.getTrackbarPos('preFilterSize','disp')*2 + 5
        textureThreshold = cv.getTrackbarPos('textureThreshold','disp')
    else:
        P1 = cv.getTrackbarPos('P1','disp')*16
        P2 = cv.getTrackbarPos('P2','disp')*32
        mode = cv.getTrackbarPos('mode','disp')
     


    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    if type(stereo) == cv.StereoBM:
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setTextureThreshold(textureThreshold)
    else:
        stereo.setP1(P1)
        stereo.setP2(P2)
        stereo.setMode(mode)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv.imshow("disp",disparity)
 
    # Close window using esc key
    if cv.waitKey(1) == 27:
      break
    cv.namedWindow("disparity", flags= cv.WINDOW_NORMAL)
    #cv.imshow("disparity", disparity)

    
print()
print()

print(f'numDisparities: {numDisparities * 16}')
print(f'blockSize: {blockSize*2 +5}')
print(f'preFilterCap: {preFilterCap}')
print(f'uniquenessRatio: {uniquenessRatio}')
print(f'speckleRange: {speckleRange}')
print(f'speckleWindowSize: {speckleWindowSize*2}')
print(f'disp12MaxDiff: {disp12MaxDiff}')
print(f'minDisparity: {minDisparity}')

if type(stereo) == cv.StereoBM:
    print(f'preFilterType: {preFilterType}')
    print(f'preFilterSize: {preFilterSize*2+5}')
    print(f'textureThreshold: {textureThreshold}')
else:
    print(f'P1: {P1*16}')
    print(f'P2: {P2*32}')
    print(f'mode: {mode}')

cv.waitKey()