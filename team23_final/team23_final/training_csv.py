#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

### Load training images and labels
imageDirectory1 = '/home/michelangelo/Documents/Tofunmi/2022Fimgs/'
imageDirectory2 = '/home/michelangelo/Documents/Tofunmi/2022Simgs/'
imageDirectory3 = '/home/michelangelo/Documents/Tofunmi/2022Fheldout/'
imageDirectory4 = '/home/michelangelo/Documents/Tofunmi/2021imgs/train_images/'
imageDirectory5 = '/home/michelangelo/ros2_ws_tofunmi/src/image_classifier-master/data/training/'

# Setting parameter values
t_lower = 90  # Lower Threshold
t_upper = 100  # Upper threshold

# Directory 1
with open(imageDirectory1 + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

train1 = np.array([np.array(cv2.imread(imageDirectory1 +lines[i][0]+".png",0)[50:258,:]) for i in range(len(lines))])
train1_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])


# Directory 2
with open(imageDirectory2 + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

train2 = np.array([np.array(cv2.imread(imageDirectory2 +lines[i][0]+".jpg",0)[50:258,:]) for i in range(len(lines))])
train2_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

# Directory 3
with open(imageDirectory3 + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

train3 = np.array([np.array(cv2.imread(imageDirectory3 +lines[i][0]+".png",0)[50:258,:]) for i in range(len(lines))])
train3_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

# Directory 4
with open(imageDirectory4 + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

train4 = np.array([np.array(cv2.imread(imageDirectory4 +lines[i][0]+".jpg",0)[50:258,:]) for i in range(len(lines))])
train4_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

all_train = np.concatenate((train1,train2,train3,train4),axis = 0)
train_new = np.zeros([all_train.shape[0],25,33])

for i in range(all_train.shape[0]):

    # blur image
    blur = cv2.medianBlur(all_train[i],25)

    # apply threshold
    ret,thresh = cv2.threshold(blur,100,255,cv2.THRESH_BINARY)

    #applying canny edge detection
    edged = cv2.Canny(thresh, t_lower, t_upper)

    train_new[i] = cv2.resize(edged,(33,25))

    #finding contours
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w>40 and h>40:
            idx+=1
            new_img = blur[y:y+h,x:x+w]
    train_new[i] = cv2.resize(new_img,(33,25))
    
# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
train_data = train_new.flatten().reshape(train_new.shape[0], 33*25)

# normalize data
train_data = StandardScaler().fit(train_data).transform(train_data.astype(float))
print(train_data.shape)

# read in training labels
train_labels = np.concatenate((train1_labels,train2_labels,train3_labels,train4_labels),axis=0)
train_labels = train_labels.reshape(train_labels.shape[0],1)
print(train_labels.shape)

# Concatenate features with corresponding labels
all_data = np.concatenate((train_data,train_labels),axis=1)
print(all_data.shape)
# Store feature vectors in csv file
with open(imageDirectory5 + 'training_data.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     writer.writerows(all_data)

