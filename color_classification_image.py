
import cv2
import color_histogram_feature_extraction
import knn_classifier
import os
import os.path
import sys

# read the test image
try:
    source_image = cv2.imread(sys.argv[1])
except:
    source_image = cv2.imread("C:/Desktop/Folder1/TestDataset/image1.jpg")
prediction = 'n.a.'

# checking whether the training data is ready
PATH = " C:/Desktop/Folder1/TRAIN"

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
prediction = knn_classifier.main('training.data', 'test.data')
print('Detected color is:', prediction)
	
