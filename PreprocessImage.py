# Preprocess the images for lane line searching
import numpy as np
import cv2

# Read the camera parameters from the previous calibration
cameraparas = np.load('cameraparas.npz')
mtx = cameraparas['mtx']
dist = cameraparas['dist']


# undistort images
def undistorting(img, mtx=mtx, dist=dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


# Extract the images features of color and gradient
def ColorGradientThreshold(img, ColThr=[150, 255], GraThr=[30, 160]):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= ColThr[0]) & (s_channel <= ColThr[1])] = 1

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=9)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= GraThr[0]) & (scaled_sobel <= GraThr[1])] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


# Define source and target matrix for perspective transformation
src = np.float32(
    [[225, 700],
     [595, 450],
     [685, 450],
     [1065, 700]]
)
dst = np.float32(
    [[250, 715],
     [250, 50],
     [980, 50],
     [980, 715]]
)


# Perform perspective transform and inverse perspective transform
def PerspectiveTransform(img, src=src, dst=dst, Tran=True):
    img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    if Tran:
        transformed = cv2.warpPerspective(img, M, img_size)
    else:
        transformed = cv2.warpPerspective(img, Minv, img_size)

    return transformed
