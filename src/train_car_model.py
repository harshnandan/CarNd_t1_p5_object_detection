import matplotlib.image as mpimg
import os
from os import walk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from multiprocessing import cpu_count
from functools import partial
from multiprocessing import Pool
from extract_feature_functions import *
from lane_processing_pipeline import *
from sklearn.model_selection import train_test_split

# Function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    # Define an empty list to receive features
    img_features = []
    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    # Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Append features to list
        img_features.append(spatial_features)
    # Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # Append features to list
        img_features.append(hist_features)
    # Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append features to list
        img_features.append(hog_features)
    # Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(windows, img, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    # Create an empty list to receive positive detection windows
    on_windows = []
    test_img = []
    # Iterate over all windows in the list
    t_feature = 0
    t_prediction = 0
    features_count = 0
    for idx, window in enumerate(windows):
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (32, 32))      
        # Extract features for that window using single_img_features()
        t1 = time.time()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        t2 = time.time()
        t_feature += t2-t1
        
        # Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # Predict using your classifier
        t1 = time.time()
        prediction = clf.predict(test_features)
        t2 = time.time()
        t_prediction += t2-t1
        
        # If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    
    print('Total Prediction time {:.5f}'.format(t_prediction))
    # Return windows for positive detections
    return on_windows
    
# Function to read all training images
def get_image_path(path, nImgs=None):
    image_path = []
    
    imgCount = 0
    for (dirpath, _, _) in walk(path):
        images = glob.glob(dirpath + '/*.png')
        for img in images:
            image_path.append(img)
            if imgCount > nImgs:
                break
            imgCount += 1
    return image_path

# Read all training data and train Support Vector Classifier (SVC)
def train_svc(params):
    # enter number of cars and notcars images to use
    nTrain_img = 8000
    # Get location of all vehicle images
    path ="../training_data/vehicles"
    cars = get_image_path(path, nTrain_img)
    vehicle_labels = [1] * len(cars)
    print('Number of vehicle images read: {}'.format(len(cars)))
    
    # Get location of all non vehicle images
    path ="../training_data/non-vehicles"
    notcars = get_image_path(path, nTrain_img)
    non_vehicle_labels = [0] * len(notcars)
    print('Number of non-vehicle images read: {}'.format(len(notcars)))
    
    ### Set parameters
    color_space = params['color_space']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    spatial_feat = params['spatial_feat']
    hist_feat = params['hist_feat']
    hog_feat = params['hog_feat']
    
#     #plot hog feature
#     sample_img = mpimg.imread(cars[52])
#     sample_img_cspace = cv2.cvtColor(sample_img, cv2.COLOR_RGB2HLS)
#      
#     _, vis_img_ch0 = get_hog_features(sample_img_cspace[:,:,0], 
#                                             orient, pix_per_cell, cell_per_block, 
#                                             vis=True, feature_vec=True)
#     _, vis_img_ch1 = get_hog_features(sample_img_cspace[:,:,1], 
#                                             orient, pix_per_cell, cell_per_block, 
#                                             vis=True, feature_vec=True)
#     _, vis_img_ch2 = get_hog_features(sample_img_cspace[:,:,2], 
#                                             orient, pix_per_cell, cell_per_block, 
#                                             vis=True, feature_vec=True)
#     plt.subplot(231)
#     plt.imshow(sample_img)
#     plt.subplot(232)
#     plt.imshow(sample_img_cspace)
#     plt.subplot(234)
#     plt.imshow(vis_img_ch0)
#     plt.subplot(235)
#     plt.imshow(vis_img_ch1)
#     plt.subplot(236)
#     plt.imshow(vis_img_ch2)
#     plt.show
    
    # Extract features from car images
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    # Extract features from notcar images
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    # Print out summary of key parameters
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC(C=0.05)
    # Check the training time for the SVC
    t1=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print('It took {:.2f} seconds to train the classifier...'.format(round(t2-t1, 2)))
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # pickle the trained model
    print('Pickling SVM model ...')
    save_quant = {'clf': svc, 'X_scaler':X_scaler}
    with open('../car_classifier.p', 'wb') as handle:
        pickle.dump(save_quant, handle)
    
    return svc, X_scaler

# Function 
def find_car_in_frame(image, svc, X_scaler, params, windows=None):

    # Set parameters
    color_space = params['color_space']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    spatial_feat = params['spatial_feat']
    hist_feat = params['hist_feat']
    hog_feat = params['hog_feat']

    # Copy the image
    draw_image = np.copy(image)

    # Training data was extracted using png images which are
    # scaled 0 to 1 by mpimg
    # image I am searching is a .jpg (scaled 0 to 255)
    draw_image = draw_image.astype(np.float32)/255
    
    # if  a window list is not already supplied
    if not windows:
        x_start_stop = [[0, 1280], [0, 1280], [0, 1280], [0, 1280], 
                        [0, 1280], [0, 1280], [0, 1280]]
        y_start_stop = [[400, 464], [416, 528], [432, 528], [400, 528], 
                        [432, 560], [400, 596], [464, 690]]
        windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                            scale=[1.0, 1.0, 1.5, 2.0, 2.0, 3.5, 3.0],  
                            xy_window=(64, 64), xy_overlap=(0.5, 0.5))
    # for debug only
    window_img = draw_boxes(draw_image, windows, color=(0, 0, 1), thick=2)
    
    # Use classifier to identify windows which have car
    t1=time.time()
    hot_windows = search_windows(windows, draw_image, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    t2=time.time() 
    print('{} windows to search for cars...'.format(len(windows) ))
    print('{:2f} seconds to find cars.'.format(t2-t1))
    
#   window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=3)
    heat_img, labels = add_heat(np.zeros(image.shape[0:2]), hot_windows, threshold=1)
    
#    marked_image, bbox = draw_labeled_bboxes(window_img, labels)
    marked_image, bbox = draw_labeled_bboxes(image, labels)
    
#     plt.figure(figsize=(12, 2))
#     plt.subplot(1,3,1)
#     plt.imshow(window_img)
#     plt.subplot(1,3,2)
#     plt.imshow(heat_img, cmap='hot')
#     plt.subplot(1,3,3)
#     plt.imshow(marked_image)
#     plt.show()
    
    return marked_image, heat_img, bbox

# Function to define image feature parameters
def get_params():
    params = {}
    params['color_space'] = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    params['orient'] = 11  # HOG orientations
    params['pix_per_cell'] = 16 # HOG pixels per cell
    params['cell_per_block'] = 2 # HOG cells per block
    params['hog_channel'] = 'ALL'    # Can be 0, 1, 2, or "ALL"
    params['spatial_size'] = (16, 16) # Spatial binning dimensions
    params['hist_bins'] = 8    # Number of histogram bins
    params['spatial_feat'] = True # Spatial features on or off
    params['hist_feat'] = True # Histogram features on or off
    params['hog_feat'] = True # HOG features on or off
    
    return params

if __name__ == '__main__':

    params = get_params()
    
    if os.path.isfile('../car_classifier.p'):
        print('Loading trained model ../car_classifier.p')
        print('To train a new model please delete this file')
        load_quant = pickle.load(open('../car_classifier.p', 'rb'))
        svc = load_quant['clf']
        X_scaler = load_quant['X_scaler']
    else:
        svc, X_scaler = train_svc(params)
    
    images_path = glob.glob('../test_images/test*.jpg')
    for img_path in images_path:
        img = mpimg.imread(img_path)
        marked_image, heat_img, bbox = find_car_in_frame(img, svc, X_scaler, params)
    
        plt.figure()
        plt.imshow(marked_image)
        plt.show()
    
