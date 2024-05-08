import os
import glob
import re


if __name__ == "__main__":
    leftPath = "./images/calibration/leftCal_test/"
    rightPath = "./images/calibration/rightCal_test/"
    os.mkdir("./images/calibration/leftCal_match/")
    os.mkdir("./images/calibration/rightCal_match/")

    left = glob.glob(leftPath+'*.png')
    for img_left in left:
        number_png = re.split('_', img_left)[3]
        result = glob.glob(rightPath + '*' +f'{number_png}')
        if(len(result) is not 0):
            img_right = result[0]
            os.rename(img_left, f"./images/calibration/leftCal_match/calibleft_out_{number_png}")
            os.rename(img_right, f"./images/calibration/rightCal_match/calibright_out_{number_png}")
