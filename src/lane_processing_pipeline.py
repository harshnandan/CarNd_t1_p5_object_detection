import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt


def load_cam_calibration():
    with open(r'../cam_calibration.p', 'rb') as f:
        cam_calibration = pickle.load(f)
        mtx = cam_calibration['mtx']
        dist = cam_calibration['dist']
        return mtx, dist

def plot_2_img(img1, title_1, img2, title_2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.tight_layout()
    
    ax1.imshow(img1)
    ax1.set_title(title_1, fontsize=20)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title_2, fontsize=20)
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.1)
    plt.show()

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude 
    sobelmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelmag/np.max(sobelmag))
    # Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # Return this mask as your binary_output image
    return sxbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    absx = np.absolute(sobelx)
    absy = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1    

    return binary_output 

def hls_thresh(image, thresh= (0,255)):
    """ write pydoc """
    # Convert image to HLS color space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Binary image mask
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    
    return binary_output 

def combined_threshld(img):
    # 
    img_xthresh = abs_sobel_thresh(img, orient='x', sobel_kernel=7, thresh=(20, 200))
    #plot_2_img(img, "Undistorted Image", img_xthresh, "X threshold")
    
    img_ythresh = abs_sobel_thresh(img, orient='y', sobel_kernel=7, thresh=(20, 200))
    #plot_2_img(img, "Undistorted Image", img_ythresh, "Y threshold")
    
    img_hlsthresh = hls_thresh(img, thresh=(120, 255))
#    plot_2_img(img_xthresh, "X-Sobel Thresholded", img_hlsthresh, "HLS Thresholded")
    
    combined_binary = np.zeros_like(img_xthresh)
    combined_binary[(img_hlsthresh==1) | (img_xthresh==1)] = 1
    #plot_2_img(img, "Undistorted Image", combined_binary, "Combined threshold")
    
    return combined_binary

def calcPerspectiveTransform(pltFlg=False):
    # read image for perspective transform
    imgLoc = '../test_images/straight_lines1.jpg'
    img_pers = cv2.imread(imgLoc)
    img_pers = cv2.cvtColor(img_pers, cv2.COLOR_BGR2RGB)
    img_height = img_pers.shape[0]
    img_width = img_pers.shape[1]
    
    src = np.array([[1100, img_height], [685  , 450], [595, 450], [200, img_height]], dtype=np.float32)
    dst = np.array([[1000, img_height], [1000, 0], [240, 0], [240, img_height]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    
    if pltFlg:
        warp_transform = perspective_transform(img_pers, M)
        
        img_pers_with_lane = cv2.polylines(img_pers, [np.int32(src)], True, (255,0,0),3)
        warp_transform_with_lane = cv2.polylines(warp_transform, [np.int32(dst)], True, (255,0,0),3)
        
        plot_2_img(img_pers_with_lane, 'Undistorted Image', warp_transform_with_lane, 'Warped Image')
    
    return M, M_inv

def perspective_transform(image, M):
    img_size = (image.shape[1], image.shape[0])
    warped_image = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_image

def window_search(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 6
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit

def lane_visualization_warped(binary_warped, left_fit, right_fit):
    # VISUALIZATION
    margin = 50
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
#     # Color in left and right line pixels
#     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
#     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
#     
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    return result

def update_lane_pos(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
            left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
            right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def calc_curvature_center(left_fit, right_fit, img_height, img_width):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_height-1, img_height )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Fit new polynomials to x,y in world space
    lane_width = right_fitx[-1] - left_fitx[-1]
    xm_per_pix = 3.7/lane_width # meters per pixel in x dimension
    
    #
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = ploty[-1]
    left_curverad = np.average((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = np.average((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    lane_curvature = np.mean([left_curverad, right_curverad])
    lane_position = np.mean([right_fitx[-1], left_fitx[-1]])
    lane_offset = (lane_position - img_width//2) * xm_per_pix
    
    calc_check = 2 > left_curverad / right_curverad > 0.5
    
    return lane_curvature, lane_offset, calc_check

def project_lane(img, left_fit, right_fit, M_inv):
    # Create new image to draw lines on
    warped_zero = np.zeros_like(img[:,:,0]).astype('uint8')
    color_warped = np.dstack((warped_zero, warped_zero, warped_zero))
    
    # Evaluate plotting points
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_x =  np.polyval(left_fit, ploty)
    right_x = np.polyval(right_fit, ploty)
    
    # Create a polygon to represent the detected lane
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warped, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warped = cv2.warpPerspective(color_warped, M_inv, (img.shape[1], img.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, new_warped, 0.3, 0)
   
    return result    

def laneMarker(img, M, M_inv, mtx, dist, counter, left_fit_prev, right_fit_prev):
    # correct camera distortion
    imgUndistort = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Get the binary image 
    combined_binary = combined_threshld(img)
    
    # Apply perspective transform to the image
    unwarped = perspective_transform(combined_binary, M)
    #plot_2_img(combined_binary, "Combined threshold", unwarped, "Unwarped")
    
    # find the initial location of lane lines
    if counter == 0:
        left_fit, right_fit = window_search(unwarped)
    else:
        left_fit, right_fit = update_lane_pos(unwarped, left_fit_prev, right_fit_prev)
    
    unwarped_marked = lane_visualization_warped(unwarped, left_fit, right_fit)
    lane_curv, lane_offset, calc_check = calc_curvature_center(left_fit, right_fit, 
                                   unwarped_marked.shape[0], unwarped_marked.shape[1])
    
    return imgUndistort, left_fit, right_fit, lane_curv, lane_offset, calc_check, combined_binary, unwarped_marked


def annotate_output_figure(imgUndistort, left_fit, right_fit, combined_binary, unwarped_marked, M_inv):
    
    lane_curv, lane_offset, calc_check = calc_curvature_center(left_fit, right_fit, 
                               unwarped_marked.shape[0], unwarped_marked.shape[1])
    
    cv2.putText(imgUndistort, 'Lane Curvature: {:.0f} m'.format(lane_curv), 
                (int(imgUndistort.shape[1]/1.9), 100), cv2.FONT_ITALIC, 1.5, (255,255,255), 5)
    cv2.putText(imgUndistort, 'Vehicle Offset: {:.2f} m'.format(lane_offset), 
                (int(imgUndistort.shape[1]/1.9), 150), cv2.FONT_ITALIC, 1.5, (255,255,255), 5)

    # project lane on the undistorted image
    marked_img = project_lane(imgUndistort, left_fit, right_fit, M_inv)

    combined_binary_small = cv2.resize(combined_binary, (0,0), fx=0.2, fy=0.2)
    unwarped_small = cv2.resize(unwarped_marked, (0,0), fx=0.2, fy=0.2) 
    
    composite_img = marked_img
    composite_img[20:20+unwarped_small.shape[0], 20:20+unwarped_small.shape[1], :] = 255*np.dstack((combined_binary_small, combined_binary_small, combined_binary_small))
    composite_img[20:20+unwarped_small.shape[0], 40+unwarped_small.shape[1]:40+2*unwarped_small.shape[1], :] = unwarped_small

    return composite_img  


if __name__ == '__main__':

    # calculate perspective transform
    M, M_inv = calcPerspectiveTransform(pltFlg=False)
    
    # read image
    filename = 'straight_lines1.jpg'
#     filename = 'test5.jpg'
    imgLoc = r'../test_images/' + filename

    img = cv2.imread(imgLoc)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # read camera calibration
    mtx, dist = load_cam_calibration()
    
    # detect lane location and mark them
    imgUndistort, left_fit, right_fit, lane_curv, lane_offset, calc_check, combined_binary, unwarped_marked = \
        laneMarker(img, M, M_inv, mtx, dist, 0, np.array([0, 0, 0]), np.array([0, 0, 0]))
    
    composite_img = annotate_output_figure(imgUndistort, left_fit, right_fit, combined_binary, unwarped_marked, M_inv)
    plt.imshow(composite_img)
    plt.savefig(r'../output_images/' + filename)
    plt.show()