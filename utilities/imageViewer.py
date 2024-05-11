import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


if __name__ == "__main__":

    path = "./kittiDataSet/sequences/00"
    image = "000000.png"

    img1 = cv.imread(f'{path}/image_0/{image}', cv.IMREAD_UNCHANGED) #queryimage # left image
    img2 = cv.imread(f'{path}/image_1/{image}', cv.IMREAD_UNCHANGED) #trainimage # right image

    stereo = cv.StereoBM_create()

    stereo.setNumDisparities(80)
    stereo.setBlockSize(21)
    stereo.setPreFilterCap(11)
    stereo.setUniquenessRatio(0)
    stereo.setSpeckleRange(1)
    stereo.setSpeckleWindowSize(0)
    stereo.setDisp12MaxDiff(14)
    stereo.setMinDisparity(3)
    stereo.setPreFilterType(0)
    stereo.setPreFilterSize(17)
    stereo.setTextureThreshold(0)

    disparity = stereo.compute(img1, img2).astype(np.float32)/16

    fig, ax = plt.subplots()
    #ax.imshow(disparity)
    #plt.waitforbuttonpress()

    # Displaying the disparity map
    # disparity = cv.normalize(disparity, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
    #cv.imshow("disp", disparity)
    #cv.waitKey()

    ax.hist(np.ravel(disparity))
    plt.show()
    plt.waitforbuttonpress()

    print(np.max(disparity))
    print(np.min(disparity))
    
