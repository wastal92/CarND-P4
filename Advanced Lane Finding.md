# Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

### Camera Calibration

#### 1. Undistorting an image

The code for this part is contained in the `GetCameraParas.py`. First I prepared "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated  array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrationCamera()` funtion. I applied this distortion correction to the test image using the `cv2.undistort()` funtion and obtained this result:

![p1](https://github.com/wastal92/CarND-P4/blob/master/examples/p1.jpg)

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one: 

![p2](https://github.com/wastal92/CarND-P4/blob/master/examples/p2.jpg)

#### 2. Color and gradient thresholds for image preprocessing

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at function `ColorGradientThreshold()` in `PreprocessImage.py`). Here's an example of my output for this step:

![p3](https://github.com/wastal92/CarND-P4/blob/master/examples/p3.jpg)

#### 3. Perform a perspective transform

The code for my perspective transform includes a function called `PerspectiveTransform()` in `PreprocessImage.py`. This function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. I chose the source and destination points as following:

|Source|Destination|
|------|-----------|
|(225, 700)|(250, 715)|
|(595, 450)|(250, 50)|
|(685, 450)|(980, 50)|
|(1065, 700)|(980, 715)|

I verified that my perspective transform was working as expected by drawing the src and dst points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![p4](https://github.com/wastal92/CarND-P4/blob/master/examples/p4.jpg)

#### 4. Identify lane line pixels and fit them

I used sliding windows to search the lane line pixels on the warped image for the first time(codes located at function `slidingwindowsearch()` in `FindLaneLine.py`). Then I searched in a margin around the previous line position for the following frame of the video(line 145-155 in `FindLaneLine.py`). After each search I fit these pixels with a 2nd order polynomial and then performed sanity check including similar curvature and reasonable distance horizontally(codes found at function `fitandcheck()` in `FindLaneLine.py`). If the fit line passed the check, it would be appended to the recent 5 measurements and then take an average to obtain the lane position. But if the fit line failed, the line would droped and the measurement list remain the same. So the lane would be the same as the previous frame until a eligible fit line was found. The found pixels and the fit line in an example is like this:

![p5](https://github.com/wastal92/CarND-P4/blob/master/examples/p5.jpg)

#### 5. Calculate radius of the lane and position of the vehicle

I did this in the function `getcurveandposition()` in `FindLaneLine.py`. 

#### 6. Example of the result

I implemented this step in lines 215-247 in `FindLaneLine.py`. Here is an example of my result on a test image:

![p6](https://github.com/wastal92/CarND-P4/blob/master/examples/p6.jpg)

---

### Pipeline (video)

Here's a link to my video result.

---

### Discussion

In this project, I think the feature extraction is the essential. I tried many combination of different color channels, color thresholds and gradient type and thresholds to obtain a suitable binary image. But when I applied the method that succeeded in the project video on the challenge video, it still yield a very bad result. Because the shadow in the video disturbed the extraction. So I think I need to find a more robust combination of lane line feature extraction for further research.
