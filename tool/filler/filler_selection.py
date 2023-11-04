import sys
import os
sys.path.append(os.getcwd()+"/Wav2Lip")
print(os.getcwd()+"/Wav2Lip")
import cv2
import numpy as np
import torch, face_detection
import face_alignment
import json, subprocess, random, string
# from PIL import Image
# import ffmpeg
    
class Filler():
    def __init__(self, outname):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,flip_input=False, device=self.device)
        # self.trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.filler_face_area = 0
        self.filler_vid_start = -1
        self.filler_vid_end = -1
        self.start = 80
        self.end = 90
        self.outname = outname
        print('Using {} for inference.'.format(self.device))

    def detect_faces(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = min(fps*300, frame_count)
        buf = []
        temp = np.empty((frame_height, frame_width, 3), np.dtype('uint8'))
        filler_face = np.empty((frame_height, frame_width, 3), np.dtype('uint8'))
        fc = 0
        cont=0
        ret = True
        
        image_name = self.outname + '.jpg'
        video_name = self.outname + '.avi'
        while (fc < frame_count  and ret):
            ret, temp = cap.read()
            buf.append(temp)
            gray_img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            face_coordinates = self.detector.get_detections_for_batch(np.expand_dims(temp,axis=0))
            # face_coordinates = self.trained_face_data.detectMultiScale(gray_img)
            # print(cont,face_coordinates[0])
            if face_coordinates[0] is not None:
                cont += 1
                for coordinate in face_coordinates:
                    (x1, y1, x2, y2) = coordinate
                    w = x2 - x1
                    h = y2 - y1
                    area = w*h
                    if area > self.filler_face_area:
                        self.filler_face_area = area
                        filler_face = temp
                if cont > self.start and cont < self.end:
                    break
            else:
                buf = []
                cont=0

        cap.release()
        
        image = cv2.imwrite(image_name,filler_face)
        print(filler_face.shape)
        frameSize = (frame_width, frame_height)
        out = cv2.VideoWriter(str(video_name),cv2.VideoWriter_fourcc(*'DIVX'), 25, frameSize)
        for i in range(len(buf)):
            img = buf[i]
            out.write(img)
        out.release()
        return(self.filler_face_area,self.filler_vid_start,self.filler_vid_end)

w2l = Filler(sys.argv[2])

w2l.detect_faces(sys.argv[1])



























# import cv2 
   
# # path 
# path = 'L94U40J.jpg'
   
# # Reading an image in default mode
# image = cv2.imread(path)
   
# # Window name in which image is displayed
# window_name = 'Image'
# trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start coordinate, here (5, 5)
# # represents the top left corner of rectangle
# start_point = (5, 5)
  
# # Ending coordinate, here (220, 220)
# # represents the bottom right corner of rectangle
# end_point = (220, 220)
  
# # Blue color in BGR
# color = (0, 255, 0)
  
# # Line thickness of 2 px
# thickness = 3

# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# face_coordinates = trained_face_data.detectMultiScale(gray_img)
# for coordinate in face_coordinates:
#     (x, y, w, h) = coordinate
                
# # Using cv2.rectangle() method
# # Draw a rectangle with blue line borders of thickness of 2 px
# image = cv2.rectangle(image, (x,y), (x+w,y+h), color, thickness)
  

# cv2.imwrite("my.png",image)

# # cv2.imshow("lalala", img)  
# # Displaying the image 
# # cv2.imshow(window_name, image)
