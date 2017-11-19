'''
    EYW Side Project - Motion Detection

    File: yawcam_motion_detector.py
    Author: Alfredo Salazar
    Date: 11/11/17
'''
# Import the necessary libraries
import datetime, imutils, time
import cv2, urllib
import numpy as np

# Define minimum size for actual motion in image
min_area = 500

# URL to get frames of Yawcam
url = "http://192.168.0.57:8888/out.jpg"

# Initialize the first frame in the video stream
firstFrame = None

# Loop over the frames of the stream
while True:
    # Initialize unoccupied/occupied text
    text = "Unoccupied"
    
    # Use urllib to get a frame from the webcam
    while True: 
        try:
            imgResp = urllib.urlopen(url)
            break
        except IOError:
            print "Yawcam stream grabbing is slow, trying again...\nMight want to restart program.\n"

    # Convert frame into a Numpy array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
    # Decode the array to OpenCV usable format
    frame = cv2.imdecode(imgNp,-1)

    # Check if there's no image (webcam disconnected)
    if frame is None:
        print "\nError in grabbing image. Quitting program."
        break # Break out of the loop
    
    # Resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # If the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # Compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # Compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

    # Draw the text and timestamp on the frame
    cv2.putText(frame, "Room Status: {}".format(text), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Create windows
    cv2.namedWindow("Security Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Frame Delta", cv2.WINDOW_NORMAL)

    # Show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    key = cv2.waitKey(1)

    # if the ESC key is pressed, break from the lop
    if key == 27:
        break

# Destroy all the created windows
cv2.destroyAllWindows()
