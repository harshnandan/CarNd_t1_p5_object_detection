from moviepy.editor import *

clip = (VideoFileClip("../output_videos/project_video.mp4").subclip(10, 40).resize(0.3))
clip.write_gif("../output_videos/project_video.gif")