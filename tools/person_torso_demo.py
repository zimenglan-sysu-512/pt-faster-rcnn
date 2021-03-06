#!/usr/bin/env python
#-*-coding: utf8-*-
# --------------------------------------------------------
# Demo of Person & Torso Detection
# Written by Dengke Dong (02.20.2016)
# --------------------------------------------------------

"""
Demo script showing detections (person | torso) in sample images.
"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test_ori import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from per_tor_util.person_torso_func import pose4video, \
                              pose4images, pose4images_top1

import cv2
import time
import math
import caffe
import pprint
import os, sys
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def _create_dire(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def _get_test_data(in_file, l_obj_n=6):
  im_names, im_objs = [], []
  with open(in_file) as f:
    for x in f.readlines():
      info         = x.strip().split()
      imgidx  = info[0].strip()
      im_name = imgidx + im_ext
      im_names.append(im_name)

      l_info = len(info)
      if l_info <= 1:
        continue
      info = info[1:]
      assert l_info % l_obj_n == 0, "wrong input format"

      objs = []
      for idx in xrange(len(info) / l_obj_n):
        idx2   = idx * l_obj_n
        objidx = int(  info[idx2 + 0].strip())
        x1     = float(info[idx2 + 1].strip())
        y1     = float(info[idx2 + 2].strip())
        x2     = float(info[idx2 + 3].strip())
        y2     = float(info[idx2 + 4].strip())
        cls    =       info[idx2 + 5].strip().lower()
        if x1 > x2:
          x1, x2 = x2, x1
        if y1 > y2:
          y1, y2 = y2, y1
        ltuple = (objidx, x1, y1, x2, y2, cls)
        objs.append(ltuple)
      im_objs.append(objs)

  return im_names, im_objs

def _parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Faster R-CNN demo')
  parser.add_argument('--gpu', dest='gpu_id', 
                      help='GPU device id to use [0]',
                      default=0, type=int)
  parser.add_argument('--cpu', dest='cpu_mode',
                      help='Use CPU mode (overrides --gpu)',
                      action='store_true')
  parser.add_argument('--cfg_file', dest='cfg_file',
                      help='optional config file', 
                      default=None, type=str)
  parser.add_argument('--def', dest='prototxt',
                      help='prototxt file defining the network',
                      default=None, type=str)
  parser.add_argument('--caffemodel', dest='caffemodel',
                      help='model to test',
                      default=None, type=str)
  # input image | file | directory 
  parser.add_argument('--im_path', dest='im_path',
                      help='the images to test',
                      default="", type=str, required=True)
  # target classess 
  parser.add_argument('--t_cls', dest='t_cls',
                      help='need some categories for target goal, splited by `,`',
                      default=None, type=str)
  parser.add_argument('--out_dire', dest='out_dire',
                      help='the images to visualize and save',
                      default="", type=str, required=True)
  parser.add_argument('--im_ext', dest='im_ext',
                      help='read input from file',
                      default=".jpg", type=str)
  parser.add_argument('--cls_filepath', dest='cls_filepath',
                      help='the path to the classes\' file',
                      default="", type=str)
  # is_video: 1-video, 0-images
  parser.add_argument('--is_video', dest='is_video',
                      help='images or video captured by camera.. (default for images)',
                      default=0, type=int)
  # out file (torso | person)
  parser.add_argument('--out_file', dest='out_file',
                      help='Restore the results of torso | person detection',
                      default=None, type=str)
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  args = parser.parse_args()
  return args

def _init_parse():
  # Use RPN for proposals
  cfg.TEST.HAS_RPN = True  

  args = _parse_args()
  print('Called with args:')
  print(args)

  cfg_file = args.cfg_file.strip()
  if os.path.exists(cfg_file) and os.path.isfile(cfg_file):
    cfg_from_file(cfg_file)    
  print('Using config:')
  pprint.pprint(cfg)

  # Get input and output images directories
  im_ext   = args.im_ext.strip()
  im_path  = args.im_path.strip()
  out_file = args.out_file.strip()
  out_dire = args.out_dire.strip()
  _create_dire(out_dire)

  cls_filepath = args.cls_filepath.strip()
  if not os.path.exists(cls_filepath) or \
     not os.path.isfile(cls_filepath):
    raise IOError(('{:s} not found.\n').format(cls_filepath))
  with open(cls_filepath) as f:
    classes = [x.strip().lower() for x in f.readlines()]
  classes = tuple(classes)
  
  t_cls = args.t_cls
  if t_cls is None:
    t_cls = []
  else:
    t_cls = t_cls.strip().split(",")
    t_cls = [cls.strip() for cls in t_cls]
  print "\nt_cls:", t_cls, "\n"

  prototxt = args.prototxt.strip()
  if not os.path.exists(prototxt) or \
     not os.path.isfile(prototxt):
    raise IOError(('{:s} not found.\n').format(prototxt))

  caffemodel = args.caffemodel.strip()
  if not os.path.exists(caffemodel) or \
     not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\n').format(caffemodel))

  if args.cpu_mode:
    caffe.set_mode_cpu()
    print "\nUsing CPU mode.\n"
  else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id
    print "\nUsing GPU mode.\n"
  time.sleep(2)

  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  print '\n\nLoaded network {:s}\n\n'.format(caffemodel)

  print "t_cls:", t_cls
  print "classes:", classes
  print "im_path:", im_path
  print "out_dire:", out_dire
  time.sleep(2)

  return net, args, classes, im_path, out_dire, out_file, t_cls, im_ext

if __name__ == '__main__':
  """Demo for human or torso detection"""
  net, args, classes, im_path, out_dire, out_file, t_cls, im_ext = _init_parse()
  
  # Warmup on a dummy image
  im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
  for i in xrange(2):
    im_detect(net, im)

  # #####################################################
  # begin
  # #####################################################
  if args.is_video != 0:
    print "\n\nDection of Person | Torso in Video"
    time.sleep(2)
    pose4video(net, classes, t_cls)
  else:
    print "\n\nDection of Person | Torso in images"
    time.sleep(2)
    pose4images_top1(net, classes, im_path, t_cls, out_dire, out_file)
  
  print "\n\nDetection has been done.\n\n"
