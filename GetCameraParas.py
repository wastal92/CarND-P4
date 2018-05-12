# Callulate the camera calibration parameters and saved in a npz file
import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt

images = glob.glob('.\\camera_cal\\calibration*.jpg')
nx = 9
ny = 6
objpoints = []
imgpoints = []

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

for img in images:
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        #img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        #plt.imshow(img)
        #plt.show()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez('cameraparas', mtx = mtx, dist = dist)