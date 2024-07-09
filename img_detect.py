import cv2
import numpy as np

# default called trackbar function
def setValues(x):
    print("")

# Creating the trackbars needed for adjusting the marker colour
cv2.namedWindow("Colour detectors")
cv2.createTrackbar("Upper Hue","Colour detectors",153,180,setValues)
cv2.createTrackbar("Upper Saturation","Colour detectors",255,255,setValues)
cv2.createTrackbar("Upper Value","Colour detectors",255,255,setValues)
cv2.createTrackbar("Lower Hue","Colour detectors",64,180,setValues)
cv2.createTrackbar("Lower Saturation","Colour detectors",72,255,setValues)
cv2.createTrackbar("Lower Value","Colour detectors",49,255,setValues)

# Capture the input frame from webcam
def get_frame(cap, scaling_factor):
    ret, frame=cap.read()
    #Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.9
    # Iterate until the user presses ESC key
    while True:
        frame = get_frame(cap, scaling_factor)

        #Convert the HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        u_hue = cv2.getTrackbarPos("Upper Hue", "Colour detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Colour detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Colour detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Colour detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Colour detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Colour detectors")

        # Define colour range in HSV colorspace
        Upper_hsv = np.array([u_hue,u_saturation,u_value])
        Lower_hsv = np.array([l_hue,l_saturation,l_value])

        #Threshold the HSV image to get only selected color
        mask = cv2.inRange(hsv,Lower_hsv,Upper_hsv)
        #Bitwise AND mask and original image
        res=cv2.bitwise_and(frame,frame,mask=mask)
        res=cv2.medianBlur(res,5)
        cv2.imshow('Original image', frame)
        cv2.imshow('Colour detector',res)
        #Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c==27:
            break
    cv2.destroyAllWindows()

