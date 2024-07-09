import numpy as np 
import cv2 

# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 

# Start a while loop 
while(1): 
	
	# Reading the video from the webcam in image frames 
	_, imageFrame = webcam.read() 

	# Convert the imageFrame in BGR(RGB color space) to HSV(hue-saturation-value) color space 
	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

	

	# Set range for green color and define mask 
	dark_blue_lower = np.array([48, 83, 131], np.uint8) 
	dark_blue_upper = np.array([69, 127, 193], np.uint8) 
	dark_blue_mask = cv2.inRange(hsvFrame, dark_blue_lower, dark_blue_upper) 

	# Set range for blue color and define mask 
	blue_lower = np.array([45, 161, 189], np.uint8) 
	blue_upper = np.array([97, 192, 213], np.uint8) 
	blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
	kernel = np.ones((5, 5), "uint8") 
	
	
	
	# For dark blue color 
	dark_blue_mask = cv2.dilate(dark_blue_mask, kernel) 
	res_dark_blue = cv2.bitwise_and(imageFrame, imageFrame, 
								mask = dark_blue_mask) 
	
	# For blue color 
	blue_mask = cv2.dilate(blue_mask, kernel) 
	res_blue = cv2.bitwise_and(imageFrame, imageFrame, 
							mask = blue_mask) 

	
	# Creating contour to track dark blue color 
	contours, hierarchy = cv2.findContours(dark_blue_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 800): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), 
									(x + w, y + h), 
									(0, 255, 0), 2) 
			
			cv2.putText(imageFrame, "Day Scholar", (x, y), 
						cv2.FONT_HERSHEY_SIMPLEX, 
						1.0, (255, 255, 255)) 

	# Creating contour to track blue color 
	contours, hierarchy = cv2.findContours(blue_mask, 
										cv2.RETR_TREE, 
										cv2.CHAIN_APPROX_SIMPLE) 
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 300): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), 
									(x + w, y + h), 
									(255, 0, 0), 2) 
			
			cv2.putText(imageFrame, "Hosteller", (x, y), 
						cv2.FONT_HERSHEY_DUPLEX, 
						1.0, (255, 255, 255)) 
			
	# Program Termination 
	cv2.imshow("color classifier", imageFrame) 
	if cv2.waitKey(10) & 0xFF == ord('q'): 
		webcam.release() 
		cv2.destroyAllWindows() 
		break
