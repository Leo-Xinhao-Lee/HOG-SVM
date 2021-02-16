from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
from NMS import non_max_suppression_hand
from ImagePyramid import pyramid_hand
from SlidingWindow import sliding_window
import argparse

"""
This script is for testing and visualizing the result of a single image
"""

save_folder_name = 'test'

#Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

#%%
# Upload the saved svm model:
model = joblib.load('./model/hard_model.npy')

# Test the trained classifier on an image below!
scale = 0
detections = []
# read the image you want to detect the object in:
img_path = './person.png'
img= cv2.imread(img_path)

# Try it with image resized if the image is too big
img= cv2.resize(img,(300,200)) # can change the size to default by commenting this code out our put in a random number

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.5
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=1.5, multichannel=True): # loop over each layer of the image that you take!
#for resized in pyramid_hand(img, windowSize=windowSize, downscale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)):
        # if the window does not meet our desired window size, ignore it!
        if window.shape[0] != winH or window.shape[1] !=winW: # ensure the sliding window has met the minimum size requirement
            continue
        window=color.rgb2gray(window)
        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2')  # extract HOG features from the window captured
        fds = fds.reshape(1, -1) # reshape the image to make a silouhette of hog --> feature vector
        pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window
        
        if pred == 1:
            if model.decision_function(fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
    
clone = resized.copy()

img_pre_NMS = img.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(img_pre_NMS, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
cv2.imshow("Raw Detections Before NMS", img_pre_NMS)
cv2.imwrite('./img/{}/img_pre_NMS.png'.format(save_folder_name),img_pre_NMS)
cv2.waitKey(0)

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
sc = [score[0] for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
#pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
pick, _ = non_max_suppression_hand(rects, probs = sc, overlapThresh = 0.3, method='fast')
        
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
    cv2.rectangle(img_pre_NMS, (xA, yA), (xB, yB), (0,255,0), 2)
cv2.imshow("img_post_NMS.png", img)
cv2.imwrite('./img/{}/img_post_NMS.png'.format(save_folder_name),img)
cv2.waitKey(0)

cv2.imshow("Detection results.png", img_pre_NMS)
cv2.imwrite('./img/{}/img_pre_post_NMS.png'.format(save_folder_name),img_pre_NMS)
cv2.waitKey(0)

with open('./img/{}/log.txt'.format(save_folder_name), 'w') as f:
    f.write('Total candidate boxes: {}\n'.format(len(detections)))
    f.writelines('After NMS left: {}'.format(pick.shape[0]))

