# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 12:29:37 2014 by @author: Guillermo
http://docs.opencv.org/trunk/doc/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
Edited on Fri Jun 1, 2018 12:24:24 by Sewify
"""
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

imgx = "la1.jpg"
sourceImg = cv2.imread(imgx)
# Source image data
rows, cols, ch = sourceImg.shape
pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])  # 4 points from the source image for the transformation

maskThreshold = 10

while True:
    # Capture frame-by-frame
    ret, img = cap.read()
    # Convert the vid to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.findchessboardCorners will find all the corners within the 9x6 box
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If corners are found, process:
    if ret:
        # Relative chessboard pts (POS of 4 of the 54 corners: 1(0), 8(9), 54(53), 46(45)) for the perspective transform
        pts2 = np.float32([corners[0, 0], corners[8, 0], corners[len(corners) - 1, 0], corners[len(corners) - 9, 0]])
        # cv2.findchessboardCorners will return a 3D array, one of which is empty. Thus the 0.

        # Compute the transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        rows, cols, ch = img.shape  # detected image data
        # make the perspective change to wrap the image to the size of the camera input
        dst = cv2.warpPerspective(sourceImg, M, (cols, rows))

        # A mask is created for adding the two images
        # maskThreshold is a variable because that allows to subtract the black background from different images
        ret, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), maskThreshold, 1, cv2.THRESH_BINARY_INV)

        # Erode and dilate are used to delete noise
        mask = cv2.erode(mask, (3, 3))
        mask = cv2.dilate(mask, (3, 3))

        # The two images are added using the mask
        for c in range(0, 3):
            img[:, :, c] = dst[:, :, c] * (1 - mask[:, :]) + img[:, :, c] * mask[:, :]
    #        cv2.imshow('mask',mask*255)
    # finally the result is presented
    out.write(img)
    cv2.imshow('img', img)

    # Wait for the key to quit
    key = cv2.waitKey(1)
    if key == ord('q'):  # quit
        print('Quit')
        break

# When everything is done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
