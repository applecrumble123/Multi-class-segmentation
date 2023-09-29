# Program To Read video
# and Extract Frames

import cv2
import os
from PIL import Image

# Function to extract frames
def FrameCapture(vid_path, saved_frame_folder):
	# Path to video file
    vidObj = cv2.VideoCapture(vid_path)
    #vidObj.set(cv2.CAP_PROP_FPS, 0.01)
    
    # Used as counter variable
    count = 0
    new_count = 0
    # checks whether frames were extracted
    success = 1
    
    while success:
        #vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        # 1 fps
        # original is 10 fps
        if count == 0 or count % 10 == 0: 
            
            if success == False:
                break
            #converted = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            #print(converted.shape)
            #pil_im = Image.fromarray(converted)
            #print(success)
            # Saves the frames with frame-count
            #img = cv2.imread(image)
            img_path = os.path.join(saved_frame_folder, "{}.png".format(new_count))
            #pil_im.save(img_path)
            cv2.imwrite(img_path,image)
            #img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            #scale_percent = 220 # percent of original size
            #width = int(img.shape[1] * scale_percent / 100)
            #height = int(img.shape[0] * scale_percent / 100)
            #dim = (width, height)
            #resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            #cv2.imwrite(img_path,resized_img)
            new_count = new_count + 1
        count += 1
    
        #if count == 300:
            #break
    
    print("Number of frames: {}".format(count))
       
root_video_folder_path = "/home/johnathon/Desktop"		

root_frames_folder_path = "/home/johnathon/Desktop/"

vid_path = os.path.join(root_video_folder_path, "Cam5.avi")

saved_frames_folder_path = os.path.join(root_frames_folder_path, "Cam5_frames_folder")



# Calling the function
FrameCapture(vid_path, saved_frames_folder_path)

