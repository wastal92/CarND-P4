# Process the images to find lane line 
# main function is used to output the result video
import numpy as np
import cv2
from LineClass import Line
from moviepy.editor import VideoFileClip
from PreprocessImage import undistorting, ColorGradientThreshold, PerspectiveTransform

# Define left and right lane lines
LeftLine = Line()
RightLine = Line()


# Process the images and find the lane lines
def ProcessImage(image):
    # Preprocess the images
    image = undistorting(image)
    img = ColorGradientThreshold(image)
    binary_warped = PerspectiveTransform(img)

    # Get the shape of images
    x_size, y_size = binary_warped.shape[1], binary_warped.shape[0]

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the width of the windows +/- margin
    margin = 100
    # Set the n iterations for average smoothing
    n_iteration = 5
    # Initialize searching status
    reset_search = True

    # Get radius and position of the lane line in real world sapce
    def getcurveandposition(ploty, Lx, Ly, Rx, Ry):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(Ly * ym_per_pix, Lx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(Ry * ym_per_pix, Rx * xm_per_pix, 2)
        # Calculate the new radius of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * ploty * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * ploty * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

        # Calculate the position of the line in meters
        left_position = left_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + left_fit_cr[1] * (y_eval * ym_per_pix) + \
                        left_fit_cr[2]
        right_position = right_fit_cr[0] * (y_eval * ym_per_pix) ** 2 + right_fit_cr[1] * (y_eval * ym_per_pix) + \
                         right_fit_cr[2]
        # Calculate the offset of the vehicle to the lane center
        offset = (right_position + left_position) / 2 - x_size / 2 * xm_per_pix

        return left_curverad, right_curverad, left_position, right_position, offset

    # Sliding windows searching
    def slidingwindowsearch():
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[y_size // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(x_size // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(y_size // nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = y_size - (window + 1) * window_height
            win_y_high = y_size - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        return left_lane_inds, right_lane_inds

    # Fit the line using fit function and test the sanity of the result
    def fitandcheck(left_lane_inds, right_lane_inds):
        # Extract left and right line pixel positions
        LeftLine.allx = nonzerox[left_lane_inds]
        LeftLine.ally = nonzeroy[left_lane_inds]
        RightLine.allx = nonzerox[right_lane_inds]
        RightLine.ally = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(LeftLine.ally, LeftLine.allx, 2)
        right_fit = np.polyfit(RightLine.ally, RightLine.allx, 2)

        ploty = np.linspace(0, y_size - 1, y_size)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Calculate the new radius of curvature and horizontal position
        left_curverad, right_curverad, left_position, right_position, _ = getcurveandposition(
            ploty=ploty, Lx=LeftLine.allx, Ly=LeftLine.ally, Rx=RightLine.allx, Ry=RightLine.ally)

        # Check similar curvature
        cr_max = np.max(np.absolute(left_curverad - right_curverad))
        cr_min = np.min(np.absolute(left_curverad - right_curverad))
        check_cr = True if (cr_max - cr_min) < 2.5 else False
        # check distance horizontally
        check_dis = True if (right_position - left_position) < 4.5 else False
        # check result
        check = True if (check_cr and check_dis) else False

        if check == False:
            return check, None, None
        else:
            return check, left_fitx, right_fitx

    # In the case that the line found in previous image
    if LeftLine.detected and RightLine.detected:
        left_lane_inds = ((nonzerox > (LeftLine.best_fit[0] * (nonzeroy ** 2) + LeftLine.best_fit[1] * nonzeroy +
                                       LeftLine.best_fit[2] - margin)) & (
                                      nonzerox < (LeftLine.best_fit[0] * (nonzeroy ** 2) +
                                                  LeftLine.best_fit[1] * nonzeroy + LeftLine.best_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (RightLine.best_fit[0] * (nonzeroy ** 2) + RightLine.best_fit[1] * nonzeroy +
                                        RightLine.best_fit[2] - margin)) & (
                                       nonzerox < (RightLine.best_fit[0] * (nonzeroy ** 2) +
                                                   RightLine.best_fit[1] * nonzeroy + RightLine.best_fit[2] + margin)))

        check, LeftLine.current_x, RightLine.current_x = fitandcheck(left_lane_inds=left_lane_inds,
                                                                     right_lane_inds=right_lane_inds)
        # If the new line pass the sanity check
        if check:
            LeftLine.detected = RightLine.detected = True
        # If the new line failed the sanity check, do sliding windows search again
        else:
            left_lane_inds, right_lane_inds = slidingwindowsearch()
            check, LeftLine.current_x, RightLine.current_x = fitandcheck(left_lane_inds=left_lane_inds,
                                                                         right_lane_inds=right_lane_inds)
            # If the new line pass the sanity check
            if check:
                LeftLine.detected = RightLine.detected = True
                # Set the reset search to True
                reset_search = True
            else:
                LeftLine.detected = RightLine.detected = False
                # Set the reset search to False
                reset_search = False

    # In case that the line not found in previous image
    else:
        # Start new sliding windows search
        left_lane_inds, right_lane_inds = slidingwindowsearch()
        check, LeftLine.current_x, RightLine.current_x = fitandcheck(left_lane_inds=left_lane_inds,
                                                                     right_lane_inds=right_lane_inds)
        # If the new line pass the sanity check
        if check:
            LeftLine.detected = RightLine.detected = True
            # Set the reset search to True
            reset_search = True
        else:
            LeftLine.detected = RightLine.detected = False
            # Set the reset search to False
            reset_search = False

    # In the case that the recent fit list contains n frames of video and new lines are found in current frame
    if len(LeftLine.recent_xfitted) == n_iteration and reset_search:
        # Delete the first element in each list
        del LeftLine.recent_xfitted[0]
        del RightLine.recent_xfitted[0]

    if reset_search:
        # Append the new found lines
        LeftLine.recent_xfitted.append(LeftLine.current_x)
        RightLine.recent_xfitted.append((RightLine.current_x))

    # If the new lines are not found in the current frame, the list keep the same.
    # We still use the previous result to fit the new image
    # Calculate the average x over the last n frames
    LeftLine.bestx = np.mean(np.array(LeftLine.recent_xfitted), axis=0)
    RightLine.bestx = np.mean(np.array(RightLine.recent_xfitted), axis=0)

    # Fit the new lane line using the average x
    ploty = np.linspace(0, y_size - 1, y_size)
    LeftLine.best_fit = np.polyfit(ploty, LeftLine.bestx, 2)
    RightLine.best_fit = np.polyfit(ploty, RightLine.bestx, 2)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([LeftLine.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([RightLine.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Inverse perspective transform of the line image
    newwarp = PerspectiveTransform(color_warp, Tran=False)
    # Put the line and the original image together
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # Write radius and offset on the image
    # Calculate the radius and offset
    LeftLine.radius_of_curvature, RightLine.radius_of_curvature, _, _, offset = getcurveandposition(
        ploty=ploty, Lx=LeftLine.bestx, Ly=ploty, Rx=RightLine.bestx, Ry=ploty)
    Radius = (LeftLine.radius_of_curvature[int(np.max(ploty))] + RightLine.radius_of_curvature[int(np.max(ploty))]) / 2

    cv2.putText(result, 'Radius of Curvature = %d(m)' % (Radius), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (255, 255, 255), thickness=2)
    if offset > 0:
        cv2.putText(result, 'Vehicle is %.2f(m) left of center' % (np.absolute(offset)), (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'Vehicle is %.2f(m) right of center' % (np.absolute(offset)), (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (255, 255, 255), thickness=2)

    return result


if __name__ == '__main__':
    white_output = 'project_video_result.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    # clip1 = VideoFileClip("project_video.mp4").subclip(0,3)
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(ProcessImage)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
