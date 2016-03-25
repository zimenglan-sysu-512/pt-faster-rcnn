#!/usr/bin/env python

import os
import cv2
import sys
import time

vis_colors = [
    (222,  12,  39), 
    (23,   12, 216), 
    (122, 212, 139), 
    (20,  198,  68), 
    (111,  12, 139), 
    (131, 112, 179), 
    (61,  211, 119), 
    (31,  131, 192), 
    (172,  51,  92), 
    (192,  21, 212), 
    (23,  119, 188), 
    (216, 121,  92), 
    (116,   11, 62), 
    (16,  111, 162), 
    (96,   46,  12), 
]

def file2im_paths(in_file, obj_n=6):
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
      assert l_info % obj_n == 0, "wrong input format"

      objs = []
      for idx in xrange(len(info) / obj_n):
        idx2   = idx * obj_n
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

def im_paths(im_path):
  im_path = im_path.strip()
  if os.path.isfile(im_path):
  	# just an image (with other image extension?)
    if im_path.endswith(".jpg") or im_path.endswith(".png") \
        or im_path.endswith(".jpeg"): 
      im_paths = [im_path]
    # read from label file: contain im_path [label ...]
    else: 
      im_paths, _ = file2im_paths(im_path)
  # read from image directory
  elif os.path.isdir(im_path):  
    im_names = os.listdir(im_path)
    assert len(im_names) >= 1
    # sort it for some convinience
    im_names.sort() 
    im_paths = [im_path + im_name.strip() for im_name in im_names]
  else:
    raise IOError(('{:s} not exist').format(im_path))

  im_n = len(im_paths)
  assert im_n >= 1, "invalid input of `im_path`: " % (im_path,)

  im_names = [os.path.basename(im_path) for im_path in im_paths]
  assert im_n == len(im_names)

  return im_paths, im_names
