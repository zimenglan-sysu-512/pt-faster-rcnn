#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np

# url:
# 		http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

im_path = "/home/ddk/dongdk/demo.jpg"
im      = cv2.imread(im_path)
im      = cv2.resize(im, None, fx=0.5, fy=0.5, \
										 interpolation=cv2.INTER_LINEAR)
# im      = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# h, w = im.shape # rows, cols 
# im = cv2.flip(im, 1)
h, w, _ = im.shape

angle = -5
p     = (w / 2, h / 2)
scale = 1	
M     = cv2.getRotationMatrix2D(p, angle, scale)

cv2.imshow("demo", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

im2   = cv2.warpAffine(im, M, (w, h))
# cv2.imshow("rotated demo", im2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

M   = np.float32([[1, 0, -10],[0, 1, 5]])
im3 = cv2.warpAffine(im, M, (w, h))

cv2.imshow("translate demo", im3)
cv2.waitKey(0)
cv2.destroyAllWindows()