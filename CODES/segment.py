# organize imports
import cv2
import imutils
import numpy as np

# global variables
bg = None
#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# MAIN FUNCTION
#-----------------
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
        
        #cv2.imshow("original",frame)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
        #print(frame.shape)
        #tuned settings
        lowerBoundary = np.array([0,40,30],dtype="uint8")
        upperBoundary = np.array([43,255,254],dtype="uint8")

        skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)

        # apply a series of erosions and dilations to the mask using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        lowerBoundary = np.array([170,80,30],dtype="uint8")
        upperBoundary = np.array([180,255,250],dtype="uint8")

        skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
        skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
        #print(skinMask.flatten())
        #print(skinMask.shape)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.medianBlur(skinMask, 5)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)
        frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
        skin = cv2.bitwise_and(frame, frame, mask = skinMask)

        cv2.imshow("masked",skin) # Everything apart from skin is shown to be black
        
        h,w = skin.shape[:2]
        bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)  # Convert image from HSV to BGR format
        bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)  # Convert image from BGR to gray format
        bw_image = cv2.GaussianBlur(bw_image,(5,5),0)  # Highlight the main object
        threshold = 1
        cv2.imshow("Before thresholded",bw_image)
        for i in range(h):
            for j in range(w):
                if bw_image[i][j] > threshold:
                    bw_image[i][j] = 0
                else:
                    bw_image[i][j] = 255
        # display the frame with segmented hand
        cv2.imshow("Video Feed", bw_image)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()