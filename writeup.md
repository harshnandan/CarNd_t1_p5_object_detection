
---

## Vehicle Detection Project

**The goals / steps of this project are the following:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply necessary transforms and use relevant features to features. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Apply strategies to filter out noisy detections.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image1]: ./img/training_data_set.png
[image2]: ./img/hog_feature.png
[image3]: ./img/sliding_window.png
[image4]: ./img/heat_map_bbox.png
[image5]: ./img/video_frame_with_significance_test.png
[video1]: ./output_videos/project_video.gif



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The core function which extracts HOG is implemented in ‘get_hog_features' within ‘extract_feature_functions.py' (line #8 to #29). This function is in turn called by ‘extract_features' in the same file. This function calls the ‘get_hog_features' with right parameters. The function also calls ‘bin_spatial' and ‘color_hist' functions to compute additional features which are then used to create an augmented feature vector.

The flow starts by training a 'LinearSVC' classifier, which uses pre-labeled images to compute the set of training set. The process starts with reading all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:


![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the ‘HSV’ color space and HOG parameters of ‘orientations=11’, ‘pixels_per_cell=(16, 16)’ and ‘cells_per_block=(2, 2)’:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found that I get high test and cross-validation accuracy with HOG parameters of ‘orientations=11’, ‘pixels_per_cell=(16, 16)’ and ‘cells_per_block=(2, 2)’.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Each training image was reduced to (32x32x3) as opposed to their original size of (64x64x3) has a augmented feature vector including spatial bins, color histogram and HOG feature vector. This way a single image was represented by a feature vector which is has a length of 1980. 

I first attempted to use an SVC classifier with rbf kernel using around 1600 (8000 vehicle and 8000 non-vehicle) images with ‘C=0.1’ and ‘Gamma=0.05’. But the training took too long and the cross-validation result were not as good. Considering the amount of time, it took to classify (and also prediction time was higher) I switched to ‘LinearSVC’ and experimented with various values of C (miss-classification cost). A ‘C=0.05’ provided the optimum result for my case.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Since the amount of computing is directly proportional to the number of search windows a tight control was maintained over windows of various scale. The smallest windows were of size (64x64) and other window sizes were defined by scaling the window with a factor.

| x_start | x_end | y_start | y_end | scale | overlap |
|:-------:|:-----:|:-------:|:-----:|:-------:|:-----:|
|  0  | 1280 | 400 | 464 | 1.0 | 0.5 |
|  0  | 1280 | 416 | 528 | 1.0 | 0.5 |
|  0  | 1280 | 432 | 528 | 1.5 | 0.5 |
|  0  | 1280 | 400 | 528 | 2.0 | 0.5 |
|  0  | 1280 | 432 | 560 | 2.0 | 0.5 |
|  0  | 1280 | 400 | 596 | 3.5 | 0.5 |
|  0  | 1280 | 464 | 690 | 3.0 | 0.5 |


![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately, I searched on five scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an accurate results.  To filter out false positives results found using various sliding windows were overlaped to create a heatmap and then it was thresholded to drop the noise. Here is an example image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/PbUlLF5n0uo)

A low quality gif of the output from the video pipeline is shown here.

![alt text][video1]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

A significance presence test was used to filter out false positive. In this strategy a running log of last 20 images were maintained and then they were overlapped using ‘add_heat’ function to come up with overlap of all 20 frames of data. Then this image was thresholded such that the overlapping images should appear in at least 18 frames. 

### Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The video stream here has cars which are visible in good lighting condition. If the lighting condition is not good (example night time or too much glare due to sun or due to snow) or the images are distorted due to rain drops on the wind-shield the pipeline may just fall apart due to problem in extracting feature vectors. These cases will need qaugmentation fro different kinds of sensors like LIDAR. Also the processing is not real-time. With current pipeline 2-3 frames are processed every second. For actual implementation we need 25 frames/sec processing.

