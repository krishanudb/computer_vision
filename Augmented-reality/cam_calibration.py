#Intrinsic Camera Calibration Code
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import sys

image_location = sys.argv[1]

#Chessboard rows and columns
rows = 3
cols = 5

#Termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

#ObjectPoints
Gridsize = 30 #(in mm) 
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
objectPointsScaled = objectPoints*30

objectPointsArray = []
imgPointsArray = []

img_all = glob.glob(image_location + "/*.jpg")

for path in img_all:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
    if ret:
        # Corner position refinement
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objectPointsArray.append(objectPointsScaled)
        imgPointsArray.append(corners)
        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

# print('objpts',objectPointsArray)
# print('imgpts',imgPointsArray)

# Camera calibration (Intrinsic)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
np.savez('mobile_calibration/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

#Finding the calibration error by reprojection
error = 0
for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)
print("Total error: ", error / len(objectPointsArray))

# New camera calibration matrix for undistorting the image
img = cv2.imread('mobile_calibration/120.jpg')
h, w = img.shape[:2]
newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistortedImg = cv2.undistort(img, mtx, dist, None, newCameraMtx)
newCameraMtx = np.array(newCameraMtx, dtype = "double")
print('camera matrix = ',newCameraMtx)
