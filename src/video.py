# Import everything needed to edit/save/watch video clips
import os
from moviepy.editor import VideoFileClip
from lane_processing_pipeline import *
import numpy as np
from train_car_model import *

class image_processor_class():
    def __init__(self, clip, outputFile, params):
        videoClip = clip
        
        # variables used for smoothing
        self.timeStepCounter = 0
        self.left_fit_prev = np.array([0, 0, 0])
        self.right_fit_prev = np.array([0, 0, 0])
        self.av_window = 1
        self.av_window_limit = 5
        self.valid_estimate = np.zeros((self.av_window_limit, 1))
        self.weight_arr = np.zeros((self.av_window_limit, 3))
        self.left_ma_arr = np.zeros((self.av_window_limit, 3))
        self.left_fit_av = np.zeros((1, 3))
        self.right_ma_arr = np.zeros((self.av_window_limit, 3))
        self.right_fit_av = np.zeros((1, 3))
        self.params = params
        
        # process video clip
        white_clip = videoClip.fl_image(self.process_image) 
        white_clip.write_videofile(outputFile, audio=False)        
    
    def process_image(self, img):

        if os.path.isfile('../car_classifier.p'):
            print('Loading trained model ../car_classifier.p')
            print('To train a new model please delete this file')
            load_quant = pickle.load(open('../car_classifier.p', 'rb'))
            svc = load_quant['clf']
            X_scaler = load_quant['X_scaler']
        else:
            svc, X_scaler = train_load_svc(self.params)
        
        marked_image, heat_img = find_car_in_frame(img, svc, X_scaler, self.params)
        
#         plt.subplot(121)
#         plt.imshow(marked_image)
#         plt.subplot(122)
#         plt.imshow(heat_img, cmap='hot')
#         plt.show()
        return marked_image
    
        
if __name__ == '__main__':
    
    #fileList = glob.glob("../*.mp4")
    fileList = [r'project_video.mp4']
    
    params = {}
    params['color_space'] = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    params['orient'] = 11  # HOG orientations
    params['pix_per_cell'] = 8 # HOG pixels per cell
    params['cell_per_block'] = 2 # HOG cells per block
    params['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
    params['spatial_size'] = (16, 16) # Spatial binning dimensions
    params['hist_bins'] = 8    # Number of histogram bins
    params['spatial_feat'] = True # Spatial features on or off
    params['hist_feat'] = True # Histogram features on or off
    params['hog_feat'] = True # HOG features on or off
    params['y_start_stop'] = [400, 670] # Min and max in y to search in slide_window()
    
    # iterate over all files
    for figIdx, fileName in enumerate(fileList):
    
        inputFile = '../' + fileName
        outputFile = '../output_videos/' + fileName
        print(inputFile)
        
        clip1 = VideoFileClip(inputFile).subclip(16, 16.6)
#         clip1 = VideoFileClip(inputFile)
        
        # process video clip
        oImageProc = image_processor_class(clip1, outputFile, params)