# Importing the necessary modules:

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *
from SlidingWindow import sliding_window

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3


# define path to images:

pos_im_path = "E:/datasets/INRIAPerson/Train/pos" # This is the path of our positive input dataset
# define the same for negatives
neg_im_path= "E:/datasets/INRIAPerson/Train/neg"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) # simply states the total no. of images
num_neg_samples = size(neg_im_listing)
print("positive: {}".format(num_pos_samples)) # prints the number value of the no.of samples in positive dataset
print("negative: {}".format(num_neg_samples))
data= []
labels = []

# compute HOG features and label them:

for file in pos_im_listing: #this loop enables reading the files in the pos_im_listing variable one by one
    img = Image.open(pos_im_path + '\\' + file) # open the file
    img = img.resize((64,128))
    gray = img.convert('L') # convert the image into single channel i.e. RGB to grayscale
    # calculate HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) # fd= feature descriptor
    data.append(fd)
    labels.append(1)

posData = np.stack(data)
posLabels = np.array(labels)

# Same for the negative images
for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    img = img.resize((64,128))
    gray= img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#%%
# Partitioning the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
print(" Constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	np.array(data), labels, test_size=0.20, random_state=42)

#%% Train the linear SVM
print(" Training Linear SVM classifier...")
model = LinearSVC()
model.fit(trainData, trainLabels)

#%% Evaluate the classifier
print(" Evaluating classifier on test data(pre-mining) ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

#%% Hard Negative Mining
hard_data = []
false_pos_prob = []
hard_labels = []
(winW, winH)= (64,128)
windowSize=(winW,winH)
downscale=1.5
if os.path.exists('./data/hard_data.npy'):
    hard_data = np.load('./data/hard_data.npy')             # save the hard negative data to save time
    hard_labels = np.zeros(hard_data.shape[0], dtype=int)
else:
    for file in neg_im_listing:
        img= Image.open(neg_im_path + '\\' + file)
        img = img.resize((200,300))

        # Now we calculate the HOG for negative features
        scale = 0
        for resized in pyramid_gaussian(img, downscale=1.5, multichannel=True):
            for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it!
                if window.shape[0] != winH or window.shape[1] != winW:  # ensure the sliding window has met the minimum size requirement
                    continue
                window = color.rgb2gray(window)
                fd = hog(window, orientations, pixels_per_cell, cells_per_block,
                          block_norm='L2')  # extract HOG features from the window captured
                fd = fd.reshape(1, -1)  # reshape the image to make a silouhette of hog --> feature vector
                pred = model.predict(fd)  # use the SVM model to make a prediction on the HOG features extracted from the window

                if pred == 1:
                    hard_data.append(fd)
                    false_pos_prob.append(model.decision_function(fd))
                    hard_labels.append(0)
            scale += 1

    idx = np.argsort(np.array([false_pos_prob[i][0] for i in range(len(false_pos_prob))]))
    idx = list(idx)
    idx.reverse()   # false-pos prob descending
    hard_data = np.concatenate(hard_data)   # hard_data: list of vector -> array
    hard_data = hard_data[idx]
    hard_labels = le.fit_transform(hard_labels)

# np.save('./data/hard_data', hard_data)

print("Total Hard negative: {}".format(hard_labels.shape[0]))   # Total Hard negative: 97043
retrainData = np.concatenate((posData, hard_data[:1200]))
retrainLabels = np.concatenate((posLabels, hard_labels[:1200]))
(retrainData, testData, retrainLabels, testLabels) = train_test_split(retrainData, retrainLabels, test_size=0.20, random_state=42)
model.fit(retrainData, retrainLabels)


#%% Evaluate the classifier
print(" \nEvaluating classifier on test data(post-mining) ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))

# Save the model:
#%% Save the Model
joblib.dump(model, 'hard_model.npy')
