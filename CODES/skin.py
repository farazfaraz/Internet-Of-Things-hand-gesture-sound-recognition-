import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

if __name__ == "__main__":
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        
        hsv_hand = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        cv2.imshow("original",hsv_hand)
        
        lowerBoundary = np.array([0,40,30],dtype="uint8")
        upperBoundary = np.array([43,255,254],dtype="uint8")
        skinMask = cv2.inRange(hsv_hand, lowerBoundary, upperBoundary)
        cv2.imshow("original",skinMask)
        result = cv2.bitwise_and(frame, frame, mask=skinMask)
        cv2.imshow("original",result)


        
        
        
        
        
        
        
        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()