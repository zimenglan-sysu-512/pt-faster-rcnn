#!/usr/bin/env python
#-*-coding: utf8-*-
# --------------------------------------------------------
# Demo of Person & Torso Detection
# Written by Dengke Dong (02.20.2016)
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test_ori import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from per_tor_util.person_torso_func import *
import caffe

import cv2
import time
import math
import os, sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

per_tor_dxy = 10

# origin as faster-rcnn using plt for showing image
def _demo4image(net, im_path, classes, t_cls, out_dire=None, NMS_THRESH = 0.3, CONF_THRESH = 0.8):
  """Detect object classes in an image using pre-computed object proposals."""
  timer = Timer()
  timer.tic()
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  scores, boxes = im_detect(net, im)
  
  # ignore bg
  is_target = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind   += 1 
    cls_boxes  = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep       = nms(dets, NMS_THRESH)
    dets       = dets[keep, :]
    vis_detections(im, cls, dets, im_path, out_dire, thresh=CONF_THRESH)

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

# using cv2 for showing image
def _demo4image_2(net, im_path, classes, t_cls, out_dire=None, NMS_THRESH = 0.3, CONF_THRESH = 0.6, im_ext=".jpg"):
  """Detect object classes in an image using pre-computed object proposals."""
  timer = Timer()
  timer.tic()
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  scores, boxes = im_detect(net, im)

  # ignore bg
  is_target = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind   += 1 
    cls_boxes  = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep       = nms(dets, NMS_THRESH)
    dets       = dets[keep, :]
    inds       = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
      continue
    for i in inds:
      bbox  = dets[i, :4]
      bbox  = [int(b) for b in bbox]
      score = dets[i, -1]
      p1    = (bbox[0], bbox[1])
      p2    = (bbox[2], bbox[3])
      cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
      p3    = (bbox[0], bbox[1] - 5)
      cv2.putText(im, '{:s} {:s}'.format(cls, str(score),), p3, \
          cv2.FONT_HERSHEY_SIMPLEX, .56, (123, 19, 208), 1)
  
  im_name = im_path.rsplit("/", 1)[1]
  im_name = im_name.rsplit(".", 1)[0]
  if out_dire and len(out_dire) > 0:
    im_path2 = out_dire + im_name + im_ext
    cv2.imwrite(im_path2, im)
  else:
    cv2.imshow(im_name, im)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

# like _demo4image_2, just show the top 1 (highest score) bbox for each class without nms
def _demo4image_top1(net, im_path, classes, t_cls, out_dire=None, NMS_THRESH = 0.3, CONF_THRESH = 0.6, im_ext=".jpg"):
  """Detect object classes in an image using pre-computed object proposals."""
  timer = Timer()
  timer.tic()
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  scores, boxes = im_detect(net, im)

  # ignore bg
  h, w, _   = im.shape
  is_target = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind       += 1 
    cls_scores     = scores[:, cls_ind]
    order          = cls_scores.argsort()[::-1]
    obj_ind        = order[0]
    score          = cls_scores[obj_ind]
    bbox           = boxes[obj_ind, 4 * cls_ind: 4 * (cls_ind + 1)]
    bbox           = [int(b) for b in bbox]
    # 
    x1, y1, x2, y2 = bbox
    # x1             = x1 - per_tor_dxy
    # y1             = y1 - per_tor_dxy
    # x2             = x2 + per_tor_dxy
    # y2             = y2 + per_tor_dxy
    # # modify by hand
    # x1             = max(x1, 1)
    # y1             = max(y1, 1)
    # x2             = min(x2, w - 2)
    # y2             = min(y2, h - 2)
    bbox           = [x1, y1, x2, y2]
    # 
    cls   = classes[cls_ind]
    p1    = (bbox[0], bbox[1])
    p2    = (bbox[2], bbox[3])
    cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
    p3    = (bbox[0], bbox[1] - 5)
    cv2.putText(im, '{:s} {:s}'.format(cls, str(score),), p3, \
        cv2.FONT_HERSHEY_SIMPLEX, .56, (123, 19, 208), 1)
      
  im_name = im_path.rsplit("/", 1)[1]
  im_name = im_name.rsplit(".", 1)[0]
  if out_dire and len(out_dire) > 0:
    im_path2 = out_dire + im_name + im_ext
    cv2.imwrite(im_path2, im)
  else:
    cv2.imshow(im_name, im)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

# for single category of multi-instances like person or torso, each image each line
def _demo4image2file(net, im_path, fhd, classes, t_cls, NMS_THRESH = 0.3, CONF_THRESH = 0.8):
  """
  Detect object classes in an image using pre-computed object proposals.
  And write the results of bboxes into file by file handler `fhd` object.

  Format:
    im_path [[score bbox cls] ...]
  """
  timer = Timer()
  timer.tic()

  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  scores, boxes = im_detect(net, im)
  
  # ignore bg
  h, w, _   = im.shape
  info      = im_path.strip()
  is_target = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind   += 1 
    cls_boxes  = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep       = nms(dets, NMS_THRESH)
    dets       = dets[keep, :]
    inds       = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
      return
    for i in inds:
      score          = dets[i, -1]
      score          = str(score)
      bbox           = dets[i, :4]
      bbox           = [int(b) for b in bbox]
      # 
      x1, y1, x2, y2 = bbox
      # x1             = x1 - per_tor_dxy
      # y1             = y1 - per_tor_dxy
      # x2             = x2 + per_tor_dxy
      # y2             = y2 + per_tor_dxy
      # # modify by hand
      # x1             = max(x1, 1)
      # y1             = max(y1, 1)
      # x2             = min(x2, w - 2)
      # y2             = min(y2, h - 2)
      bbox           = [x1, y1, x2, y2]
      # 
      bbox           = [str(b) for b in bbox]
      bbox           = " ".join(bbox).strip()
      info           = info  + " " + score + " " + bbox + " " + cls
    fhd.write(info.strip() + "\n")

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

# for single category of top 1 instance like person or torso, each image each line
def _demo4image2file_top1(net, im_path, fhd, classes, t_cls, NMS_THRESH = 0.3, CONF_THRESH = 0.8):
  """
  Detect object classes in an image using pre-computed object proposals.
  And write the results of bboxes into file by file handler `fhd`

  Format:
    im_path [[score bbox cls] ...]
  """
  timer = Timer()
  timer.tic()
  print 'Demo for {}'.format(im_path)
  im            = cv2.imread(im_path)
  scores, boxes = im_detect(net, im)

  # ignore bg
  h, w, _       = im.shape
  info          = im_path.strip()
  is_target     = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind       += 1 
    cls_scores     = scores[:, cls_ind]
    order          = cls_scores.argsort()[::-1]
    obj_ind        = order[0]
    score          = cls_scores[obj_ind]
    score          = str(score)
    bbox           = boxes[obj_ind, 4 * cls_ind: 4 * (cls_ind + 1)]
    bbox           = [int(b) for b in bbox]
    # 
    x1, y1, x2, y2 = bbox
    # x1             = x1 - per_tor_dxy
    # y1             = y1 - per_tor_dxy
    # x2             = x2 + per_tor_dxy
    # y2             = y2 + per_tor_dxy
    # # modify by hand
    # x1             = max(x1, 1)
    # y1             = max(y1, 1)
    # x2             = min(x2, w - 2)
    # y2             = min(y2, h - 2)
    bbox           = [x1, y1, x2, y2]
    # 
    bbox           = [str(b) for b in bbox]
    bbox           = " ".join(bbox).strip()
    info           = info + " " + score + " " + bbox + " " + cls
  # for all categories in one line
  fhd.write(info.strip() + "\n")

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

# not save the results (visualization)
def _demo4video(net, im, classes, t_cls, NMS_THRESH=0.3, CONF_THRESH=0.8):
  """Detect object classes in an image using pre-computed object proposals."""
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  
  # ignore bg
  is_target = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind   += 1 
    cls_boxes  = boxes[:, 4 * cls_ind: 4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets       = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
    keep       = nms(dets, NMS_THRESH)
    dets       = dets[keep, :]

    # draw bboxes
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
      cv2.imshow("frame", im)
      return

    for i in inds:
      bbox = dets[i, :4]
      bbox = [int(b) for b in bbox]
      score = dets[i, -1]
      # bbox
      x1 = bbox[0]
      y1 = bbox[1]
      x2 = bbox[2]
      y2 = bbox[3]
      # draw
      p1 = (x1, y1)
      p2 = (x2, y2)
      cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
      p3 = (x1, y1 - 3)
      cv2.putText(im, '{:s} {:.3f}'.format(cls, score), p3, \
          cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))
    # show for all categories
    cv2.imshow("frame", im)

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

def _im_paths(im_path):
  im_path = im_path.strip()
  if os.path.isfile(im_path):
    if im_path.endswith(".jpg") or im_path.endswith(".png") \
        or im_path.endswith(".jpeg"): # # just an image (with other image extension?)
      im_paths = [im_path]
    else: # read from label file: contain im_path [label ...]
      im_paths, _ = _get_test_data(im_path)
  elif os.path.isdir(im_path):  # read from image directory
    im_names = os.listdir(im_path)
    assert len(im_names) >= 1
    im_names.sort() # sort it for some convinience
    im_paths = [im_path + im_name.strip() for im_name in im_names]
  else:
    raise IOError(('{:s} not exist').format(im_path))

  im_n = len(im_paths)
  assert im_n >= 1, "invalid input of `im_path`: " % (im_path,)

  return im_paths, im_n

# ##############################################################################
# 
# ##############################################################################

def pose4video(net, classes, t_cls):
  # Init camera
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print 'No camera found'
    sys.exit(1) 
  im_c = 1
  while(True):
    ret, im = cap.read()
    if im is not None and im.shape[0] != 0:
      print "Processing %s image..." % (im_c,)
      _demo4video(net, im, classes, t_cls)
      im_c = im_c + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

def pose4images(net, classes, im_path, t_cls, out_dire, out_file):
  '''process each image: person & torso detection'''
  assert os.path.isdir(out_dire) == True, "please be sure `out_dire`: %s" % (out_dire,)
  
  im_paths, im_n = _im_paths(im_path) # input images (test)

  timer = Timer()
  timer.tic()

  # write the results into file, if `out_file` is given
  if out_file and len(out_file) > 0:
    out_path = out_dire + out_file
    fhd      = open(out_path, "w")

    print "\nwrite the results into `%s`\n\n" % (out_path,)
    time.sleep(2)

    for im_c in xrange(im_n):
      print "\n\nim_c: %s (%s)" % (im_c, im_n)
      im_path = im_paths[im_c]
      _demo4image2file(net, im_path, fhd, classes, t_cls)

    fhd.close()
  else: # only show the highest-score bbox of each category
    print "\nshow the results using images"
    time.sleep(2)

    for im_c in xrange(im_n):
      print "\n\nim_c: %s (%s)" % (im_c, im_n)
      im_path = im_paths[im_c]
      _demo4image_top1(net, im_path, classes, t_cls, out_dire)

  timer.toc()
  total_time = timer.total_time
  aver_time  = total_time / (im_n + 0.)
  print "Detection took %ss for %s images (average time: %s)" % (total_time, im_n, aver_time)
