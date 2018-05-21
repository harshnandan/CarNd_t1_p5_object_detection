# Import everything needed to edit/save/watch video clips
import os
from moviepy.editor import VideoFileClip
from lane_processing_pipeline import *
import numpy as np
from train_car_model import *
from extract_feature_functions import *
import random
#from src.train_car_model import svc

class image_processor_class():
    def __init__(self, clip, outputFile, params, svc, X_scaler):
        videoClip = clip
        
        # variables used for smoothing
        self.timeStepCounter = 0
        self.left_fit_prev = np.array([0, 0, 0])
        self.right_fit_prev = np.array([0, 0, 0])
        self.av_window = 1
        self.av_window_limit = 20
        self.bbox = []
        self.params = params
        self.svc = svc
        self.X_scaler = X_scaler
        
        # process video clip
        white_clip = videoClip.fl_image(self.process_image) 
        white_clip.write_videofile(outputFile, audio=False)        
    
    def process_image(self, img):
#         cv2.imwrite('../test_images/bbox_example_10_{}.jpg'.format(self.timeStepCounter), 
#                     cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        draw_img = np.copy(img)
        #draw_image = cv2.blur(draw_img,(5, 5))
        marked_image, heat_img, bboxes = find_car_in_frame(draw_img, self.svc, self.X_scaler, self.params)

#         plt.imshow(marked_image)
#         plt.show()
        
##         do a local search around intitally indetified 
#         bboxes_lcl = []
#         for bbox in bboxes:
#             bboxes_lcl.append(bbox)
#             for ind in range(30):
#                 x_rand = random.randint(bbox[0][0], bbox[1][0])
#                 y_rand = random.randint(bbox[0][1], bbox[1][1])
#                 width = np.abs(bbox[0][0]- bbox[1][0])
#                 height = np.abs(bbox[0][1]- bbox[1][1])
#                 x1 = max([0, x_rand-width])
#                 y1 = max([0, y_rand-height])
#                 x2 = min([1280, x_rand+width])
#                 y2 = min([720, y_rand+height])
#                 bboxes_lcl.append(((x1, y1), (x2, y2)))
#         
#         draw_img = np.copy(img)
# 
#         marked_image, heat_img, bboxes_final = find_car_in_frame(draw_img, self.svc, self.X_scaler, self.params, bboxes_lcl)
# 
#         plt.imshow(marked_image)
#         plt.show()
        bboxes_final = bboxes
        
#         return marked_image
        
        time_heatMap, lbl = self.moving_average(bboxes_final, marked_image.shape[0:2])
        if not time_heatMap==None:
            filtered_image, _ = draw_labeled_bboxes(img, lbl)
#             plt.subplot(121)
#             plt.imshow(filtered_image)
#             plt.subplot(122)
#             plt.imshow(time_heatMap, cmap='hot')
#             plt.show()
            return filtered_image
        else:
            return img
        

    def moving_average(self, bbox, imgSize):
        # before window_limit is reached
        if self.timeStepCounter > self.av_window_limit:
            heatImg = np.zeros(imgSize)
            if len(self.bbox):
                self.bbox.pop(0)
            self.bbox += bbox
            heatImg, lbl = add_heat(heatImg, self.bbox, threshold=np.int(self.av_window_limit*0.9))
        else:
            self.bbox += bbox
            heatImg = None
            lbl = None
        
        self.timeStepCounter += 1
        return heatImg, lbl
            
if __name__ == '__main__':
    
    #fileList = glob.glob("../*.mp4")
    fileList = [r'project_video.mp4']
    
    params = get_params()
#     params = {}
#     params['color_space'] = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#     params['orient'] = 11  # HOG orientations
#     params['pix_per_cell'] = 16 # HOG pixels per cell
#     params['cell_per_block'] = 2 # HOG cells per block
#     params['hog_channel'] = 'ALL'    # Can be 0, 1, 2, or "ALL"
#     params['spatial_size'] = (16, 16) # Spatial binning dimensions
#     params['hist_bins'] = 8    # Number of histogram bins
#     params['spatial_feat'] = True # Spatial features on or off
#     params['hist_feat'] = True # Histogram features on or off
#     params['hog_feat'] = True # HOG features on or off
#     params['y_start_stop'] = [400, 670] # Min and max in y to search in slide_window()


    if os.path.isfile('../car_classifier.p'):
        print('Loading trained model ../car_classifier.p')
        print('To train a new model please delete this file')
        load_quant = pickle.load(open('../car_classifier.p', 'rb'))
        svc = load_quant['clf']
        X_scaler = load_quant['X_scaler']
    else:
        svc, X_scaler = train_load_svc(self.params)
            
    # iterate over all files
    for figIdx, fileName in enumerate(fileList):
    
        inputFile = '../' + fileName
        outputFile = '../output_videos/' + fileName
        print(inputFile)
        
        clip1 = VideoFileClip(inputFile).subclip(0, 50)
#         clip1 = VideoFileClip(inputFile)
        
        # process video clip
        oImageProc = image_processor_class(clip1, outputFile, params, svc, X_scaler)