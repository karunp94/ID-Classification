
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cumfreq

import knn_classifier as knn_classifier


def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    with open('test.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_training_image(img_name):

    # detect image color by using image file name to label training data
    if 'LightBlue' in img_name:
        data_source = 'LightBlue'
    
    elif 'Darkblue' in img_name:
        data_source = 'Darkblue'
    
    elif 'NoID' in img_name:
        data_source = 'NoID'
    

    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('training.data', 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training():

    
    for f in os.listdir('./TRAIN/Lightblue'):
        color_histogram_of_training_image('./TRAIN/LightBlue/' + f)

    for f in os.listdir('./TRAIN/Darkblue'):
        color_histogram_of_training_image('./TRAIN/Darkblue/' + f)		
    
    for f in os.listdir('./TRAIN/NoID'):
        color_histogram_of_training_image('./TRAIN/NoID/' + f)
