'''
    EYW Side Project - Motion Detection

    File: webcam_motion_detector.py
    Author: Alfredo Salazar
    Date: 11/16/17
'''
# Import the necessary libraries
import datetime, imutils, time, cv2
import numpy as np

# Define minimum size for actual motion in image
min_area = 500

# Initialize webcam videocapture object
cap = cv2.VideoCapture(0)

# Initialize the first frame in the video stream
firstFrame = None

# Initialize counting variable
count = 0

# Loop over the frames of the stream
while True:
    # Initialize unoccupied/occupied text
    text = "Unoccupied"

    # Read frame from webcam
    ret, frame = cap.read()

    # Check if there is no image
    if ret == None:
        print "\nError in grabbing image. Quitting program."
        break # Break out of the loop.
    
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

    ret2, firstTest = cap.read()

    firstTest = imutils.resize(firstTest, width=500)
    testGray = cv2.cvtColor(firstTest, cv2.COLOR_BGR2GRAY)
    testGray = cv2.GaussianBlur(testGray, (21, 21), 0)

    def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err
    '''
    testDst = cv2.compare(testGray, gray, cv2.CMP_EQ)

    if testDst.all() == 255:
        count = count + 1
        if count == 10:
            firstFrame = testGray
            count = 0
    '''
    
# Destroy all the created windows
cv2.destroyAllWindows()

# Release the webcam object
cap.release()
