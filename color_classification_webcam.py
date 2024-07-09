
import cv2
import color_histogram_feature_extraction
import color_classification_webcam1
import knn_classifier
import os
import os.path

cap = cv2.VideoCapture(0)
(ret, frame) = cap.read()
prediction = 'n.a.'

# checking whether the training data is ready
PATH = './training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

while True:

    (ret, frame) = cap.read()

    
    cv2.putText(
        frame,               # The image (frame) onto which text will be drawn
        'Prediction: ' + prediction,  # The text string to be drawn
        (15, 45),            # The position (x, y) coordinates of the text baseline on the image
        cv2.FONT_HERSHEY_PLAIN,  # The font type used for drawing the text (change to a different type)
        3,                   # Font scale factor that is multiplied by the font-specific base size
        (255, 255, 255),     # Text color (in this case, a bright color like white)
        lineType=cv2.LINE_AA, # Optional: Line type for the text (anti-aliased line)
        thickness=5
        )

    # Display the resulting frame
    cv2.imshow('color classifier', color_classification_webcam1.imageFrame)

    color_histogram_feature_extraction.color_histogram_of_test_image(frame)

    prediction = knn_classifier.main('training.data', 'test.data')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()		
