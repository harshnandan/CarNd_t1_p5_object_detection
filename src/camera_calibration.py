import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_2_img(img1, img2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.tight_layout()
    
    ax1.imshow(img1)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(img2)
    ax2.set_title('Undistorted Image', fontsize=20)
    
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def calibrateImg(imgList, nx, ny):
    
    # Go through all calibration image    
    for imgLoc in imgList:
        print(imgLoc)
        img = cv2.imread(imgLoc)
        # Convert to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # get corner of chess board, ret=1 if all corners are found
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        print(ret)
        # if corners found
        if ret:
            imgpoints.append(corners)
            
            cornerMarkedImg = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            
            objpts = np.zeros((nx*ny, 3), dtype = np.float32)
            objpts[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
            objpoints.append(objpts)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2][::-1], None, None)        
    return mtx, dist, cornerMarkedImg

# Read in an image
CalImgLocation = r'../camera_cal/*.jpg'
imgList = glob.glob(CalImgLocation)
imgpoints = []
objpoints = []
nx = 9;
ny = 6;

# find corners and caliberate image
mtx, dist, _ = calibrateImg(imgList, nx, ny)
cam_calibration = {'mtx':mtx, 'dist':dist}
pickle.dump(cam_calibration, open(r'..\cam_calibration.p', 'wb'))

# select a samaple image to test undistortion
img = cv2.imread(r'../camera_cal/calibration4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
cornerMarkedImg = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

unDistortedImg= cv2.undistort(cornerMarkedImg, mtx, dist, None, mtx)

# plot the sample undistortion  
plot_2_img(cornerMarkedImg, unDistortedImg)


