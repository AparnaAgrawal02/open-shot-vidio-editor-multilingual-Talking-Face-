''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np


# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if haarcascade is in data directory
    if (haarcascade in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

# get image from webcam
#print ("checking webcam for connection ...")
#webcam_cap = cv2.VideoCapture(0)


i = 0
while i == 0:
    # read webcam
    #_, frame = webcam_cap.read()
    # frame = cv2.imread('p.png')
    frame = cv2.imread('../received/Frame-00001.png')
    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(frame)
    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(gray)
    
    pts = []
    # print(faces)
    for (x,y,w,d) in faces:
        # Detect landmarks on "gray"
        print(gray.shape, faces.shape)
        _, landmarks = landmark_detector.fit(gray, np.array(faces))
        print(faces, x, y, w, d, landmarks)
        # print(len(landmarks[0][0]))
        for landmark in landmarks:
            # print(len(landmarks[0]))
            for i, (x,y) in enumerate(landmark[0]):
                if i > 26:
                    break
                # print(x, y)
                pts.append([int(x), int(y)])
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 2
                # cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)

    # print(len(pts))
    pts = np.array(pts)
    print(pts)
    pts[17:] = pts[17:][::-1]
    # print(pts[17:])
    # print(pts)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = frame[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    # inverted_mask = 255 - mask
    # print(mask.shape, frame.shape, croped.shape)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst

    cv2.imwrite("croped.png", croped)
    cv2.imwrite("mask.png", mask)
    cv2.imwrite("dst.png", dst)
    cv2.imwrite("dst2.png", dst2)
    # save last instance of detected image
    # cp = frame.copy()
    
    frame_copy = np.copy(frame).astype(np.float)
    print(frame_copy)
    frame_copy[y:y+h, x:x+w][:, :, 0] = frame_copy[y:y+h, x:x+w][:, :, 0] - mask
    frame_copy[y:y+h, x:x+w][:, :, 1] = frame_copy[y:y+h, x:x+w][:, :, 1] - mask
    frame_copy[y:y+h, x:x+w][:, :, 2] = frame_copy[y:y+h, x:x+w][:, :, 2] - mask
    # print(frame_copy[frame_copy < 0])
    frame_copy[frame_copy < 0] = 0
    cv2.imwrite('face-detect2.jpg', frame_copy)    
    
    # Show image
    #cv2.imshow("frame", frame)
    i += 1
    # terminate the capture window
    if cv2.waitKey(20) & 0xFF  == ord('q'):
        webcam_cap.release()
        cv2.destroyAllWindows()
        break
