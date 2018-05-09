## Writeup

Some pieces of the code has been borrowed from Udacity labs.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./img/undistort_chessboard.png "Undistorted"
[image2]: ./img/undistorted_image.png "Road Undistorted"
[image3]: ./img/various_thresholding.png "Binary Example"
[image4]: ./img/threshold_image.png "Binary combined"
[image5]: ./img/warped_image.png "Warped Image with Visual Fit"
[image6]: ./img/warped_binary.png "Warped Binary Image"
[image7]: ./img/lane_pixel_histogram.png "Histogram"
[image8]: ./img/window_search.png "Window Search"
[image9]: ./img/2nd_order_equation.png "Second order polynomial"
[image10]: ./img/2nd_order_equation_curvature.png "curvature"
[image11]: ./img/lane_marked_perspective_view.png "output of pipeline"

[image12]: ./output_videos/project_video.gif "Marked Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

It is essential to correct for any distortions introduced by camera. A standalone script 'camera_calibration.py’ computes the cameta matrix 'mtx’ and the distortion coefficient 'dist’ and 'pickle’ these entities into 'cam_calibration.p’. The script works through number of calibration images supplied with the problem. The script identifies the corners within the chess board image and computes the camera matrix and distortion coefficient to map them to real world undistorted coordinates provided by the user. The result of applying camera distortion can be clearly seen in the image below.

The camera calibration function provided within openCV requires specification of 'objpoints’, which are the (x, y, z) coordinates of the chessboard corners in the world. As the chessboard is 2D the 'z’ coordinate can be assumed to be '0’. 'objpoints’ is appended to a list every time all the corners are detected on the image. 'imgpoints’ will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

'ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)'

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

'unDistortedImg= cv2.undistort(cornerMarkedImg, mtx, dist, None, mtx)’

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion Corrected Image

The pickled camera calibration file ('cam_calibration.p') saved in the previous step is reloaded into this pipeline and is used to apply correction to images taken on the road. The distortion correction is demonstrated in the image below:

![alt text][image2]

#### 2. Thresholding to Generate Binary Image

A combination of thresholding of gradient in x direction and s-channel thresholding of image in HLS space is used to generate a binary image. The code to generate the binary image is in 'combined_threshold’ function within 'image_processing_pipeline.py’. This function call more basic functions which are listed between line ’26 to 86’ in the same file.

![alt text][image3]

To combine these two thresholding methods a logical 'OR’ operator is used and the result of combined thresholding is shown below.

![alt text][image4]

#### 3. Calculating the Perspective Transform

The perspective transform is an operation which translates the perspective image into a bird’s eye view. The transform is calculated by identifying 4 points in the perspective image ('src’) and telling the OpenCV function 'getPerspectiveTransform’ about the corresponding location of these 4 points in the warped image ('dst’). For this assignment 2 points on the left lane line and 2 points on the right lane line are chosen in the undistorted image with straight lane line and because the lane lines are straight it is easy to tell the corresponding location in bird’s eye view. 

'M = cv2.getPerspectiveTransform(src, dst)'

An inverse transform can also be calculated which will transform points in warped image to points in perspective image by switching 'src’ and 'dst’.

'M = cv2.getPerspectiveTransform(dst, src)'

The code for calculating this transform is in 'calcPerspectiveTransform’ (line 105 to 127) function in 'image_processing_pipeline’.

```python
src = np.array([[1100, img_height], [685  , 450], [595, 450], [200, img_height]], dtype=np.float32)

dst = np.array([[1000, img_height], [1000, 0], [240, 0], [240, img_height]], dtype=np.float32)
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 1100, 720      | 1000, 720        | 
| 685, 450      | 1000, 0      |
| 595, 450     | 240, 0      |
| 200, 720      | 240, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


![alt text][image5]

#### 4. Lane Identification

Binary form of warped image (shown below) is used to initiate the lane finding algorithm. As the binary image is a 2D matrix containing only ones and zeros, summing the value of pixel in vertical direction indicates where the lanes are. The starting position of the lane is taken to be the midpoint of the left and right bumps in the figure below.

![alt text][image6]
![alt text][image7]

A sliding window approach is utilized to find the lane position in rest of the image. Figure below shows how the sliding window identifies the lane line in rest of image. (line number 134 to 208)

![alt text][image8]

#### 5. Radius of Curvature

A second order polynomial is fitted through the identified point and radius of curvature is calculated as follows (equations are borrowed from Udacity course material)

![alt text][image9]

![alt text][image10]

#### 6. Lane Demarcation in Perspective View

The lane line detected in warped image can be mapped back to the perspective view using the inverse transform matrix. The binary and the warped image is shown as a subfigure in the marked figure to get a sense of what is the major steps performed by the pipeline. This helps a lot to debug and understand particular behavior of the pipeline.


![alt text][image11]

---

### Pipeline (video)

The only difference for the video processing timeline and image processing timeline is that in the video processing timeline the detection of line lane in binary image from second frame onward is limited to the region next to the lane detected in previous frame. This not only makes finding the lanes faster, but it also throws out the outliers which do appear as the car is moving along the road. This also results in smoother lane detection through time.

Here's a [link to higher quality video](https://youtu.be/IUyTZYoWxmw)

![alt text][image12]

---

### Discussion

#### Limitations of current pipeline:

##### The pipeline will fail if the lane line does not start from the bottom edge of the figure.

##### Also, if a lane line disappears from the frame the lane updating algorithm will fail when the lane line reappears.  

