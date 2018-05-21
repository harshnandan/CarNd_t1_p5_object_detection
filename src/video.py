import os
from moviepy.editor import VideoFileClip
from lane_processing_pipeline import *
import numpy as np
from train_car_model import *
from extract_feature_functions import *
import random

class image_processor_class():
    def __init__(self, clip, outputFile, params, svc, X_scaler):
        videoClip = clip
        # variables used for smoothing
        self.timeStepCounter = 0
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
        # copy image
        draw_img = np.copy(img)
        # find bounding boxes for the car
        marked_image, heat_img, bboxes = find_car_in_frame(draw_img, self.svc, self.X_scaler, self.params)
#         plt.imshow(marked_image)
#         plt.show()

        bboxes_final = bboxes
        # checking for windows that appear in majority of n consicutive frames
        time_heatMap, lbl = self.significance_presence_check(bboxes_final, 
                                                marked_image.shape[0:2])
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
        

    def significance_presence_check(self, bbox, imgSize):
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
    # get feature extraction parameters
    params = get_params()
    # load car_classifier file if it is present
    # if not found train a support vector classifier
    if os.path.isfile('../car_classifier.p'):
        print('Loading trained model ../car_classifier.p')
        print('To train a new model please delete this file')
        load_quant = pickle.load(open('../car_classifier.p', 'rb'))
        svc = load_quant['clf']
        X_scaler = load_quant['X_scaler']
    else:
        svc, X_scaler = train_svc(self.params)
            
    # iterate over all video files
    for figIdx, fileName in enumerate(fileList):
        inputFile = '../' + fileName
        outputFile = '../output_videos/' + fileName
        print(inputFile)
        # load clips
        clip1 = VideoFileClip(inputFile).subclip(0, 50)
        # process video clip
        oImageProc = image_processor_class(clip1, outputFile, params, svc, X_scaler)