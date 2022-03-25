import cv2
import numpy as np

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = cv2.imread('test.jpg')
oriImage = image.copy()


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

while True:

    i = image.copy()

    if not cropping:
        cv2.imshow("image", image)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        # cv2.imshow("image", i)
        # Read image
        img = cv2.imread(i)
        hh, ww = img.shape[:2]
        # threshold on white
        # Define lower and uppper limits
        lower = np.array([200, 200, 200])
        upper = np.array([255, 255, 255])
        
        # Create mask to only select black
        thresh = cv2.inRange(img, lower, upper)

        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # invert morp image
        mask = 0 - morph
        
        # apply mask to image
        result = cv2.bitwise_and(img, img, mask=mask)
        
        
        # save results
        # cv2.imwrite('test_thresh.jpg', thresh)
        # cv2.imwrite('test_morph.jpg', morph)
        # cv2.imwrite('test_mask.jpg', mask)
        cv2.imwrite('test_result.jpg', result)
        
        cv2.imshow('thresh', thresh)
        # cv2.imshow('morph', morph)
        # cv2.imshow('mask', mask)
        #cv2.imshow('result', result)
        
        
            # cv2.waitKey(1)
        
        
   # cv2.waitKey(0)
        
    # close all open windows
    cv2.destroyAllWindows()