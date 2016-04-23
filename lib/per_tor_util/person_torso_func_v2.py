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

fps = 2
per_tor_dxy = 10

def _image2bbox(net, im, classes, t_cls, NMS_THRESH=0.3, CONF_THRESH=0.8):
  """
  Detect object classes in an image using pre-computed object proposals.
    >= threshold (maybe empty)
  """
  timer = Timer()
  timer.tic()
  scores, boxes, = im_detect(net, im)
  
  # ignore bg
  bboxes    = []
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
      break
    
    for i in inds:
      bbox = dets[i, :4]
      bbox = [int(b) for b in bbox]
      score = dets[i, -1]
      bboxes.append((bbox, score, cls))

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

  return bboxes

def _image2bbox_top1(net, im, classes, t_cls, NMS_THRESH=0.3, CONF_THRESH=0.8):
  """
  Detect object classes in an image using pre-computed object proposals.
    only highest-score one 
    (very suitable for model trained on cropped images by person detector)
    (also for model trained on whole image but maybe not good for multi-people in the image)
  """
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  
  # ignore bg
  bboxes    = []
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
    bboxes.append((bbox, score, cls))

  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

  return bboxes

def _within_bbox(cond_bbox, targ_bbox):
  '''x1, y1, x2, y2'''
  assert len(cond_bbox) == 4
  assert len(cond_bbox) == len(targ_bbox)

  return targ_bbox[0] >= cond_bbox[0] and targ_bbox[1] >= cond_bbox[1] \
     and targ_bbox[2] <= cond_bbox[2] and targ_bbox[3] <= cond_bbox[3]

def _image2bbox_top1_cond(net, im, classes, t_cls, cond_bbox, NMS_THRESH=0.3, CONF_THRESH=0.8):
  """
  Detect object classes in an image using pre-computed object proposals.

  """
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(net, im)
  
  # ignore bg
  bboxes    = []
  is_target = len(t_cls) > 0
  for cls_ind, cls in enumerate(classes[1:]):
    if  is_target and cls not in t_cls:
      continue
    cls_ind       += 1 
    cls_scores     = scores[:, cls_ind]
    order          = cls_scores.argsort()[::-1]
    for obj_ind in order:
      score          = cls_scores[obj_ind]
      bbox           = boxes[obj_ind, 4 * cls_ind: 4 * (cls_ind + 1)]
      bbox           = [int(b) for b in bbox]
      if _within_bbox(cond_bbox, bbox):
        bboxes.append((bbox, score, cls))
        break
  
  total_time = timer.toc(average=False)
  print "Detection took %ss for %s object proposals" % (total_time, boxes.shape[0])

  return bboxes

def _im_paths_from_dire(in_dire):
  im_paths = []
  dires    = os.listdir(in_dire)
  for dire in dires:
    im_path = in_dire + dire
    if not im_path.endswith("/") and not im_path.endswith(".jpg") and \
       not im_path.endswith(".jpeg") and not im_path.endswith(".png"):
      im_path = im_path + "/"
    if os.path.isdir(im_path):
      im_paths2 = _im_paths(im_path)
      if len(im_paths2) > 0:
        im_paths.extend(im_paths2)
    else:
      # print im_path
      if os.path.exists(im_path) and os.path.isfile(im_path):
        im_paths.append(im_path)

  return im_paths

def _get_test_data(in_file):
  fh = open(in_file)
  im_paths = []
  for line in fh.readlines():
    line = line.strip()
    info = line.split()

    assert len(info) >= 1
    im_path = info[0].strip()
    im_paths.append(im_path)
  fh.close()
  n_im = len(im_paths)
  assert n_im >= 1
  return im_paths, n_im

def _im_paths(im_path):
  im_path = im_path.strip()
  if os.path.isfile(im_path):
    if im_path.endswith(".jpg") or im_path.endswith(".png") \
        or im_path.endswith(".jpeg"): # # just an image (with other image extension?)
      im_paths = [im_path]
    else: # read from label file: contain im_path [label ...]
      im_paths, _ = _get_test_data(im_path)
  elif os.path.isdir(im_path):  # read from image directory
    # im_names = os.listdir(im_path)
    # assert len(im_names) >= 1
    # im_names.sort() # sort it for some convinience
    # im_paths = [im_path + im_name.strip() for im_name in im_names]
    im_paths = _im_paths_from_dire(im_path)
    assert len(im_paths) >= 1
    im_paths.sort()
  else:
    raise IOError(('{:s} not exist').format(im_path))

  n_im = len(im_paths)
  assert n_im >= 1, "invalid input of `im_path`: " % (im_path,)
  print "\n\nn_im:", n_im, "\n\n"

  return im_paths, n_im

# ####################################################################################
# 
# ####################################################################################

def init_net(net1, net2):
  # Warmup on a dummy image
  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  if net1 is not None:
    for i in xrange(2):
      im_detect(net1, im)
    print "\n\n***********************************\n\n"

  if net2 is not None:
    time.sleep(3)
    for i in xrange(2):
      im_detect(net2, im)

  print "\n\nInit Net Done!\n\n"
  time.sleep(3)

# ####################################################################################
# 
# ####################################################################################

def _bbox4images_show_v1(net, im_path, classes, t_cls, out_dire=None, threshold=0., im_ext=".jpg"):
  '''
  only highest-score person bbox with corresponding highest-score torso bbox
    (it's easy to deal with multi-people cases)
  person detector and torso detector share the same category in the classes file path
  '''
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  # detection -- only highest-score bbox
  assert len(t_cls) == 1
  bboxes = _image2bbox_top1(net, im, classes, t_cls)
  # since several categories, need to find bboxes of the target cls
  bbox             = bboxes[0]
  bbox, score, cls = bbox
  x1, y1, x2, y2   = bbox

  no_result = False
  if score < threshold:
    print "below threshold so that have no results: (%s, %s)" % (score, threshold)
    no_result = True

  if out_dire is not None:
    im_name = im_path.rsplit("/", 1)[1]
    im_name = im_name.rsplit(".", 1)[0]
    out_path = out_dire + im_name + im_ext

    if no_result:
      cv2.putText(im, 'no bbox -- {:s} {:.3f}'.format(cls, score), (20, 20), \
          cv2.FONT_HERSHEY_SIMPLEX, .6, (123, 19, 228))
    else:
      p1 = (x1, y1)
      p2 = (x2, y2)
      cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
      p3 = (x1 + 5, y2 - 5)
      cv2.putText(im, '{:s} {:.3f}'.format(cls, score), p3, cv2.FONT_HERSHEY_SIMPLEX, .6, (123, 19, 228))
      # cv2.imshow(im_name, im)
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()
    
    cv2.imwrite(out_path, im)

  if no_result:
    return None, score
  else:
    return bbox, score

def _bbox4images_show_v2(p_net, t_net, im_path, classes, t_cls, out_dire, threshold=0., im_ext=".jpg"):
  '''
  only highest-score person bbox with corresponding highest-score torso bbox
    (it's easy to deal with multi-people cases)
  person detector and torso detector share the same category in the classes file path
  '''
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  # person detection -- only highest-score bbox
  assert len(t_cls) == 1
  p_bboxes = _image2bbox_top1(p_net, im, classes, t_cls)
  # since several categories, need to find bboxes of the target cls
  p_bbox   = p_bboxes[0]
  p_bbox, p_score, p_cls = p_bbox
  p_x1, p_y1, p_x2, p_y2 = p_bbox

  if p_score < threshold:
    print "below threshold so that have no results: (%s, %s)" % (p_score, threshold)
    return

  im_name = im_path.rsplit("/", 1)[1]
  im_name = im_name.rsplit(".", 1)[0]

  # #######################################################
  # option 1: using origin image 
  # (because the when training torso detector using whole image)
  # torso detection
  # t_bboxes = _image2bbox_top1(t_net, im, classes, t_cls)
  t_bboxes = _image2bbox_top1_cond(t_net, im, classes, t_cls, p_bbox)
  t_bbox   = t_bboxes[0]
  t_bbox, t_score, t_cls = t_bbox
  t_x1, t_y1, t_x2, t_y2 = t_bbox

  # #######################################################
  # option 2: using cropped image (because the when training torso detector 
  # using cropped image detected by person detector)
  # torso detection
  # h, w, _  = im.shape
  # x1       = p_x1 - per_tor_dxy
  # y1       = p_y1 - per_tor_dxy
  # x2       = p_x2 + per_tor_dxy
  # y2       = p_y2 + per_tor_dxy
  # x1       = max(x1, 1)
  # y1       = max(y1, 1)
  # x2       = min(x2, w - 2)
  # y2       = min(y2, h - 2)
  # im2      = im[y1: y2, x1: x2]
  # t_bboxes = demo4bboxes_top1(t_net, im2, classes, t_cls)
  # t_bbox   = t_bboxes[0]
  # t_bbox, t_score, t_cls = t_bbox
  # t_x1, t_y1, t_x2, t_y2 = t_bbox
  # t_x1    += p_x1
  # t_y1    += p_y1
  # t_x2    += p_x1
  # t_y2    += p_y1

  # draw person
  p1 = (p_x1, p_y1)
  p2 = (p_x2, p_y2)
  cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
  p3 = (p_x1, p_y1 - 5)
  # cv2.putText(im, '{:s} {:.3f}'.format(p_cls, p_score), p3, cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))
  # draw torso
  p1 = (t_x1, t_y1)
  p2 = (t_x2, t_y2)
  cv2.rectangle(im, p1, p2, (138, 31, 116), 2)
  p3 = (t_x1, t_y1 - 5)
  # cv2.putText(im, '{:s} {:.3f}'.format(t_cls, t_score), p3, cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))

  # cv2.imshow(im_name, im)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  out_path = out_dire + im_name + im_ext
  cv2.imwrite(out_path, im)

def _bbox4images_show_v3(p_net, t_net, im_path, classes, t_cls, out_dire, threshold=0., im_ext=".jpg"):
  '''
  only highest-score person bbox with corresponding highest-score torso bbox
    (it's easy to deal with multi-people cases)
  person detector and torso detector share the same category in the classes file path
  '''
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  # person detection -- only highest-score bbox
  assert len(t_cls) == 1
  p_bboxes = _image2bbox_top1(p_net, im, classes, t_cls)
  # since several categories, need to find bboxes of the target cls
  p_bbox   = p_bboxes[0]
  p_bbox, p_score, p_cls = p_bbox
  p_x1, p_y1, p_x2, p_y2 = p_bbox

  if p_score < threshold:
    print "below threshold so that have no results: (%s, %s)" % (p_score, threshold)
    return

  im_name = im_path.rsplit("/", 1)[1]
  im_name = im_name.rsplit(".", 1)[0]

  # #######################################################
  # using cropped image detected by person detector)
  # torso detection
  h, w, _  = im.shape
  x1       = p_x1 - per_tor_dxy
  y1       = p_y1 - per_tor_dxy
  x2       = p_x2 + per_tor_dxy
  y2       = p_y2 + per_tor_dxy
  x1       = max(x1, 1)
  y1       = max(y1, 1)
  x2       = min(x2, w - 2)
  y2       = min(y2, h - 2)
  im2      = im[y1: y2, x1: x2]
  t_bboxes = _image2bbox_top1(t_net, im2, classes, t_cls)
  t_bbox   = t_bboxes[0]
  t_bbox, t_score, t_cls = t_bbox
  t_x1, t_y1, t_x2, t_y2 = t_bbox
  t_x1    += p_x1
  t_y1    += p_y1
  t_x2    += p_x1
  t_y2    += p_y1

  # draw person
  p1 = (p_x1, p_y1)
  p2 = (p_x2, p_y2)
  cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
  p3 = (p_x1, p_y1 - 5)
  # cv2.putText(im, '{:s} {:.3f}'.format(p_cls, p_score), p3, cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))
  # draw torso
  p1 = (t_x1, t_y1)
  p2 = (t_x2, t_y2)
  cv2.rectangle(im, p1, p2, (138, 31, 116), 2)
  p3 = (t_x1, t_y1 - 5)
  # cv2.putText(im, '{:s} {:.3f}'.format(t_cls, t_score), p3, cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))

  # cv2.imshow(im_name, im)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  
  out_path = out_dire + im_name + im_ext
  cv2.imwrite(out_path, im)

def _bbox4image2file_v1(net, im_path, classes, t_cls, fhd):
  '''
  only highest-score person bbox with corresponding highest-score torso bbox
    (it's easy to deal with multi-people cases)
  format:
    im_path [[pt_i p_x1 p_y1 p_x2 p_y2 t_x1 t_y1 t_x2 t_y2] ...]
  '''
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  assert len(t_cls) == 1
  # detection -- only highest-score bbox
  bboxes             = _image2bbox_top1(net, im, classes, t_cls)
  bbox               = bboxes[0]
  bbox, score, _     = bbox      # bbox score cls
  x1, y1, x2, y2     = bbox

  objidx = 0
  info = im_path + " " + str(objidx) + " " + str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2)
  fhd.write(info.strip() + "\n")

  return bbox, score

def _bbox4image2file_v2(p_net, t_net, im_path, classes, t_cls, fhd):
  '''
  only highest-score person bbox with corresponding highest-score torso bbox
    (it's easy to deal with multi-people cases)
  person detector and torso detector share the same category in the classes file path
  format:
    im_path [[pt_i p_x1 p_y1 p_x2 p_y2 t_x1 t_y1 t_x2 t_y2] ...]
  '''
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  assert len(t_cls) == 1
  # person detection -- only highest-score bbox
  p_bboxes = _image2bbox_top1(p_net, im, classes, t_cls)
  p_bbox   = p_bboxes[0]
  p_bbox, _, _           = p_bbox
  p_x1, p_y1, p_x2, p_y2 = p_bbox

  # #######################################################
  # option 1: using origin image (because the when training torso detector using whole image)
  # torso detection
  # t_bboxes = _image2bbox_top1(t_net, im, classes, t_cls)
  t_bboxes = _image2bbox_top1_cond(t_net, im, classes, t_cls, p_bbox)
  t_bbox   = t_bboxes[0]
  t_bbox, _, _           = t_bbox
  t_x1, t_y1, t_x2, t_y2 = t_bbox

  # #######################################################
  # option 2: using cropped image (because the when training torso detector 
  # using cropped image detected by person detector)
  # torso detection
  # h, w, _  = im.shape
  # x1       = p_x1 - per_tor_dxy
  # y1       = p_y1 - per_tor_dxy
  # x2       = p_x2 + per_tor_dxy
  # y2       = p_y2 + per_tor_dxy
  # x1       = max(x1, 1)
  # y1       = max(y1, 1)
  # x2       = min(x2, w - 2)
  # y2       = min(y2, h - 2)
  # im2      = im[y1: y2, x1: x2]
  # t_bboxes = demo4bboxes_top1(t_net, im2, classes, t_cls)
  # t_bbox   = t_bboxes[0]
  # t_bbox, _, _           = t_bbox
  # t_x1, t_y1, t_x2, t_y2 = t_bbox
  # t_x1    += p_x1
  # t_y1    += p_y1
  # t_x2    += p_x1
  # t_y2    += p_y1

  pt_i = 0
  info = im_path + " " + str(pt_i)
  info = info    + " " + str(p_x1) + " " + str(p_y1) + " " + str(p_x2) + " " + str(p_y2) \
                 + " " + str(t_x1) + " " + str(t_y1) + " " + str(t_x2) + " " + str(t_y2)
  fhd.write(info.strip() + "\n")

def _bbox4image2file_v3(p_net, t_net, im_path, classes, t_cls, fhd):
  '''
  only highest-score person bbox with corresponding highest-score torso bbox
    (it's easy to deal with multi-people cases)
  person detector and torso detector share the same category in the classes file path
  format:
    im_path [[pt_i p_x1 p_y1 p_x2 p_y2 t_x1 t_y1 t_x2 t_y2] ...]
  '''
  print 'Demo for {}'.format(im_path)
  im = cv2.imread(im_path)
  assert len(t_cls) == 1
  # person detection -- only highest-score bbox
  p_bboxes = _image2bbox_top1(p_net, im, classes, t_cls)
  p_bbox   = p_bboxes[0]
  p_bbox, _, _           = p_bbox
  p_x1, p_y1, p_x2, p_y2 = p_bbox

  # #######################################################
  # using cropped image (because the when training torso detector 
  # using cropped image detected by person detector)
  # torso detection
  h, w, _  = im.shape
  x1       = p_x1 - per_tor_dxy
  y1       = p_y1 - per_tor_dxy
  x2       = p_x2 + per_tor_dxy
  y2       = p_y2 + per_tor_dxy
  x1       = max(x1, 1)
  y1       = max(y1, 1)
  x2       = min(x2, w - 2)
  y2       = min(y2, h - 2)
  im2      = im[y1: y2, x1: x2]
  t_bboxes = _image2bbox_top1(t_net, im2, classes, t_cls)
  t_bbox   = t_bboxes[0]
  t_bbox, _, _           = t_bbox
  t_x1, t_y1, t_x2, t_y2 = t_bbox
  t_x1    += p_x1
  t_y1    += p_y1
  t_x2    += p_x1
  t_y2    += p_y1

  pt_i = 0
  info = im_path + " " + str(pt_i)
  info = info    + " " + str(p_x1) + " " + str(p_y1) + " " + str(p_x2) + " " + str(p_y2) \
                 + " " + str(t_x1) + " " + str(t_y1) + " " + str(t_x2) + " " + str(t_y2)
  fhd.write(info.strip() + "\n")

# ####################################################################################
# 
# ####################################################################################

def face4video(net, classes, t_cls, threshold=0.):
  print "Not Implemented!"

def face4images(net, classes, im_path, t_cls, out_dire, out_file, threshold=0., im_ext=".jpg"):
  is_show = os.path.isdir(out_dire)
  im_paths, im_n = _im_paths(im_path) # input images (test)

  timer = Timer() # set timer
  timer.tic()     

  results = []

  # write the results into file, if `out_file` is given
  if out_file and len(out_file) > 0:
    assert is_show == True, "please be sure `out_dire`: %s" % (out_dire,)
    out_path = out_dire + out_file
    fhd      = open(out_path, "w")
    print "\nwrite the results into `%s`\n\n" % (out_path,)

    for im_c in xrange(im_n):
      print "\nim_c: %s (%s)" % (im_c, im_n)
      im_path = im_paths[im_c]
      bbox, score = _bbox4image2file_v1(net, im_path, classes, t_cls, fhd) # bbox score
      results.append((im_path, bbox, score))

    fhd.close()
  else: # only show the highest-score bbox of each category
    print "\nshow the results using images"
    for im_c in xrange(im_n):
      timer2 = Timer()
      timer2.tic()

      print "\nim_c: %s (%s)" % (im_c, im_n)
      im_path = im_paths[im_c]
      bbox, score = _bbox4images_show_v1(net, im_path, classes, t_cls, out_dire, threshold=threshold) # bbox score
      results.append((im_path, bbox, score))
      total_time = timer2.toc(average=False)
      print "Detection took %ss for one image" % (total_time,)
      
  timer.toc()
  total_time = timer.total_time
  aver_time  = total_time / (im_n + 0.)
  print "Detection took %ss for %s images (average time: %s)" % (total_time, im_n, aver_time)

  return results

def pose4video(p_net, t_net, classes, t_cls):
  # Init camera
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print 'No camera found'
    sys.exit(1) 
  cap.set(cv2.cv.CV_CAP_PROP_FPS, fps)  # set fps

  im_c = 1
  while(True):
    ret, im = cap.read()  # read image by camera
    h, w, _ = im.shape
    im_copy = im.copy()
    if im is not None and im.shape[0] != 0:
      timer = Timer()
      timer.tic()
      print "Processing %s image..." % (im_c,)
      # person detection
      p_bboxes = _image2bbox(p_net, im_copy, classes, t_cls)

      for p_bbox in p_bboxes:
        p_bbox, p_score, p_cls = p_bbox
        p_x1, p_y1, p_x2, p_y2 = p_bbox
        # crop person
        im2                    = im_copy[p_y1: p_y2, p_x1: p_x2]  
        # torso detection
        t_bboxes = _image2bbox_top1(t_net, im2, classes, t_cls)
        # draw person
        p1 = (p_x1, p_y1)
        p2 = (p_x2, p_y2)
        cv2.rectangle(im, p1, p2, (38, 231, 16), 2)
        p3 = (p_x1, p_y1 - 5)
        cv2.putText(im, '{:s} {:.3f}'.format(p_cls, p_score), p3, cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))
        # draw torso
        for t_bbox in t_bboxes:
          t_bbox, t_score, t_cls = t_bbox
          t_x1, t_y1, t_x2, t_y2 = t_bbox
          t_x1 += p_x1
          t_y1 += p_y1
          t_x2 += p_x1
          t_y2 += p_y1
          p1    = (t_x1, t_y1)
          p2    = (t_x2, t_y2)
          cv2.rectangle(im, p1, p2, (138, 31, 116), 2)
          p3    = (t_x1, t_y1 - 5)
          cv2.putText(im, '{:s} {:.3f}'.format(t_cls, t_score), p3, cv2.FONT_HERSHEY_SIMPLEX, .36, (23, 119, 188))
      cv2.imshow("frame", im) # show for all categories
      im_c = im_c + 1
      total_time = timer.toc(average=False)
      print "Detection took %ss one image" % (total_time,)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

  # release camera
  cap.release()
  cv2.destroyAllWindows()

def pose4images(p_net, t_net, classes, im_path, t_cls, out_dire, out_file, choice=0, im_ext=".jpg"):
  '''process each image: person & torso detection'''
  assert os.path.isdir(out_dire) == True, "please be sure `out_dire`: %s" % (out_dire,)
  
  im_paths, im_n = _im_paths(im_path) # input images (test)

  timer = Timer() # set timer
  timer.tic()     

  if out_file is not None:
    out_path = out_dire + out_file
    fhd      = open(out_path, "w")
    print "\nwrite the results into `%s`\n\n" % (out_path,)

    for im_c in xrange(im_n):
      print "\nim_c: %s (%s)" % (im_c, im_n)
      im_path = im_paths[im_c]
      if choice == 0:
        _bbox4image2file_v2(p_net, t_net, im_path, classes, t_cls, fhd)
      elif choice == 1:
        _bbox4image2file_v3(p_net, t_net, im_path, classes, t_cls, fhd)
      else:
        raise ValueError("NotImplemented!")
    fhd.close()
  else: 
    print "\nshow the results using images"
    for im_c in xrange(im_n):
      timer2 = Timer()
      timer2.tic()

      print "\nim_c: %s (%s)" % (im_c, im_n)
      im_path = im_paths[im_c]
      if choice == 0:
        _bbox4images_show_v2(p_net, t_net, im_path, classes, t_cls, out_dire)
      elif choice == 1:
        _bbox4images_show_v3(p_net, t_net, im_path, classes, t_cls, out_dire)
      else:
        raise ValueError("NotImplemented!")
      
      total_time = timer2.toc(average=False)
      print "Detection took %ss for one image" % (total_time,)
      
  timer.toc()
  total_time = timer.total_time
  aver_time  = total_time / (im_n + 0.)
  print "Detection took %ss for %s images (average time: %s)" \
        % (total_time, im_n, aver_time)


# ################################
# 
# ################################



def _bbox4images_show_v21(p_net, t_net, im, classes, pt_cls):
  ''''''
  assert len(pt_cls) == 1
  p_bboxes = _image2bbox_top1(p_net, im, classes, pt_cls)
  p_bbox   = p_bboxes[0]
  p_bbox, p_score, _  = p_bbox

  t_bboxes = _image2bbox_top1_cond(t_net, im, classes, pt_cls, p_bbox)
  t_bbox   = t_bboxes[0]
  t_bbox, t_score, _  = t_bbox

  h, w, _  = im.shape
  return h, w, p_bbox, p_score, t_bbox, t_score

def _bbox4images_show_v31(p_net, t_net, im, classes, pt_cls):
  ''''''
  assert len(pt_cls) == 1
  p_bboxes = _image2bbox_top1(p_net, im, classes, pt_cls)
  p_bbox   = p_bboxes[0]
  p_bbox, p_score, _     = p_bbox
  p_x1, p_y1, p_x2, p_y2 = p_bbox

  h, w, _  = im.shape
  # x1       = p_x1 - per_tor_dxy
  # y1       = p_y1 - per_tor_dxy
  # x2       = p_x2 + per_tor_dxy
  # y2       = p_y2 + per_tor_dxy
  x1       = max(x1, 1)
  y1       = max(y1, 1)
  x2       = min(x2, w - 2)
  y2       = min(y2, h - 2)
  im2      = im[y1: y2, x1: x2]
  t_bboxes = _image2bbox_top1(t_net, im2, classes, pt_cls)
  t_bbox   = t_bboxes[0]
  t_bbox, t_score, _     = t_bbox
  t_x1, t_y1, t_x2, t_y2 = t_bbox
  t_x1    += p_x1
  t_y1    += p_y1
  t_x2    += p_x1
  t_y2    += p_y1
  t_bbox   = [t_x1, t_y1, t_x2, t_y2]

  return h, w, p_bbox, p_score, t_bbox, t_score

def pose4images_online(p_net, t_net, image, classes, pt_cls, choice=0):
  '''process each image: person & torso detection'''
  if isinstance(image, basestring) and os.path.isfile(image) and os.path.exists(image):
    im = cv2.imread(image)
  else:
    im = image.copy();

  # cv2.imshow("Demo", im)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  if choice == 0:
    res = _bbox4images_show_v21(p_net, t_net, im, classes, pt_cls)
  elif choice == 1:
    res = _bbox4images_show_v31(p_net, t_net, im, classes, pt_cls)
  else:
    raise ValueError("NotImplemented!")
  
  return res