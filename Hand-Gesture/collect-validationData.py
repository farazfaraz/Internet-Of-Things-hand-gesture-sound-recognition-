# organize imports
import cv2
import imutils
import numpy as np
import os

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
    cv2.imshow("different image", diff)

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
    # Create the directory structure
    if not os.path.exists("validData"):
        os.makedirs("validData")
        os.makedirs("validData/0")
        os.makedirs("validData/1")
        os.makedirs("validData/2")
        os.makedirs("validData/3")
        os.makedirs("validData/4")
        os.makedirs("validData/5")
   
    

    # Train or test 
    directory = 'validData/'
    
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

        # Getting count of existing images
        count = {'zero': len(os.listdir(directory+"/0")),
             'one': len(os.listdir(directory+"/1")),
             'two': len(os.listdir(directory+"/2")),
             'three': len(os.listdir(directory+"/3")),
             'four': len(os.listdir(directory+"/4")),
             'five': len(os.listdir(directory+"/5"))}
        # Printing the count in each set to the screen
        cv2.putText(frame, "MODE : "+directory, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "IMAGE COUNT", (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "ZERO : "+str(count['zero']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "ONE : "+str(count['one']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "TWO : "+str(count['two']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "THREE : "+str(count['three']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "FOUR : "+str(count['four']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        cv2.putText(frame, "FIVE : "+str(count['five']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)
        
        # observe the keypress by the user  
        #thresholded = cv2.resize(thresholded, (64, 64)) 
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27: # esc key
            break
        if interrupt & 0xFF == ord('0'):
            cv2.imwrite(directory+'0/'+str(count['zero'])+'.jpg', cv2.resize(thresholded, (64, 64)))
        if interrupt & 0xFF == ord('1'):
            cv2.imwrite(directory+'1/'+str(count['one'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('2'):
            cv2.imwrite(directory+'2/'+str(count['two'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('3'):
            cv2.imwrite(directory+'3/'+str(count['three'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('4'):
            cv2.imwrite(directory+'4/'+str(count['four'])+'.jpg', cv2.resize(thresholded, (64, 64)) )
        if interrupt & 0xFF == ord('5'):
            cv2.imwrite(directory+'5/'+str(count['five'])+'.jpg', cv2.resize(thresholded, (64, 64)) )

# free up memory
camera.release()
cv2.destroyAllWindows()