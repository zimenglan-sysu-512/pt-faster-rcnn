#!/usr/bin/env python

import os
import sys
import cv2

# url:
# 		http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

im_path = "/home/ddk/dongdk/demo.jpg"
im      = cv2.imread(im_path)
im      = cv2.resize(im, None, fx=0.5, fy=0.5, \
										 interpolation=cv2.INTER_LINEAR)
h, w, _ = im.shape # rows, cols 

angle = 15
p     = (w / 2, h / 2)
scale = 1	
M     = cv2.getRotationMatrix2D(p, angle, scale)

cv2.imshow("demo", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

im2   = cv2.warpAffine(im, M, (w, h))
cv2.imshow("rotated demo", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()