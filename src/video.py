# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from image_processing_pipeline import *
import numpy as np

class image_processor_class():
    def __init__(self, clip, outputFile):
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
        
        # process video clip
        white_clip = videoClip.fl_image(self.process_image) 
        white_clip.write_videofile(outputFile, audio=False)        
    
    def process_image(self, image):
        # calculate perspective transform
        M, M_inv = calcPerspectiveTransform()
    
        # read camera calibration
        mtx, dist = load_cam_calibration()
        
        # Find left and right line lanes
        imgUndistort, left_fit, right_fit, _, _, calc_check, combined_binary, unwarped_marked = \
            laneMarker(image, M, M_inv, mtx, dist, 0, np.array([0, 0, 0]), np.array([0, 0, 0]))
        
        self.moving_average(calc_check, left_fit, right_fit)
        
        composite_img = annotate_output_figure(imgUndistort, self.left_ma, self.right_ma, combined_binary, unwarped_marked, M_inv)
        
        self.left_fit_prev = left_fit
        self.right_fit_prev = right_fit

        self.timeStepCounter = self.timeStepCounter + 1 
        
        return composite_img
    
    def moving_average(self, calc_check, left_fit, right_fit):
        
        # before window_limit is reached
        if self.timeStepCounter < self.av_window_limit:
            if calc_check:
                self.valid_estimate[self.av_window-1] = 1
                self.left_ma_arr[self.av_window-1, :] = left_fit
                self.right_ma_arr[self.av_window-1, :] = right_fit
            else:
                self.valid_estimate[self.av_window-1] = 1
                self.left_ma_arr[self.av_window-1, :] = self.left_ma
                self.right_ma_arr[self.av_window-1, :] = self.right_ma
            self.av_window += 1
        else:
            self.left_ma_arr[0:(self.av_window_limit-1), :] = self.left_ma_arr[1:(self.av_window_limit), :]
            self.right_ma_arr[0:(self.av_window_limit-1), :] = self.right_ma_arr[1:(self.av_window_limit), :]
            self.valid_estimate[0:self.av_window_limit-1] = self.valid_estimate[1:self.av_window_limit]
            if calc_check:
                self.valid_estimate[self.av_window_limit-1] = 1
                self.left_ma_arr[self.av_window_limit-1, :] = left_fit
                self.right_ma_arr[self.av_window_limit-1, :] = right_fit
            else:
                self.valid_estimate[self.av_window_limit-1] = 1
                self.left_ma_arr[self.av_window_limit-1, :] = self.left_ma
                self.right_ma_arr[self.av_window_limit-1, :] = self.right_ma           

            
        indx = np.where(self.valid_estimate == 1)[0]
        
        weight = np.exp(range(0, -len(indx), -1) )
        sum_weight = np.sum(weight)
        weight = weight / sum_weight
        w_arr = np.vstack((weight, weight, weight))
        
        self.weight_arr[indx,:] = w_arr.T
        self.left_ma = np.sum(np.multiply(self.left_ma_arr, self.weight_arr), 0) 
        self.right_ma = np.sum(np.multiply(self.right_ma_arr, self.weight_arr), 0)  
        
if __name__ == '__main__':
    
    #fileList = glob.glob("../*.mp4")
    fileList = [r'project_video.mp4']
    
    # iterate over all files
    for figIdx, fileName in enumerate(fileList):
    
        inputFile = '../' + fileName
        outputFile = '../output_videos/' + fileName
        print(inputFile)
        
#         clip1 = VideoFileClip(inputFile).subclip(0, 5)
        clip1 = VideoFileClip(inputFile)
        
        # process video clip
        oImageProc = image_processor_class(clip1, outputFile)