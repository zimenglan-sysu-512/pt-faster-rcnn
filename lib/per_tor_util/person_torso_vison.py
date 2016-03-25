#!/usr/bin/env python
#-*-coding: utf8-*-
# --------------------------------------------------------
# Demo of Person & Torso Detection
# Written by Dengke Dong (02.20.2016)
# --------------------------------------------------------

import cv2
import math
import os, sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def vis_detections(im, cls, dets, im_path, out_dire=None, thresh=0.5, im_ext=".jpg"):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
      return

  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')

  for i in inds:
      bbox  = dets[i, :4]
      score = dets[i, -1]

      ax.add_patch(
          plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=3.5)
          )
      ax.text(bbox[0], bbox[1] - 2,
              '{:s} {:.3f}'.format(cls, score),
              bbox=dict(facecolor='blue', alpha=0.5),
              fontsize=14, color='white')

  ax.set_title(('{} detections with '
                'p({} | box) >= {:.1f}').format(cls, cls,
                                                thresh),
                fontsize=14)
  plt.axis('off')
  plt.tight_layout()

  if out_dire and len(out_dire) > 0:
    im_name  = im_path.rsplit("/", 1)[1]
    im_name  = im_name.rsplit(".", 1)[0]
    im_path2 = out_dire + im_name + im_ext
    print "\nimage path:", im_path2
    plt.savefig(im_path2)
  else:
    print "\nshow image..."
    plt.show()
 
  plt.clf()  # clear the figure handler

# difference with `vis_detections` is that how to show `title`, `cls` and `score`
def vis_detections2(im, cls, dets, im_path, out_dire=None, thresh=0.5, im_ext=".jpg"):
  """Draw detected bounding boxes."""
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
    return
  im = im[:, :, (2, 1, 0)]
  plt.imshow(im)

  # show the results as the tile form
  sub_title = "%s detections with p(%s | box) >= {:.1f}" % (cls, cls)
  plt.suptitle(sub_title)

  # draw bboxes
  for i in inds:
    bbox  = dets[i, :4]
    score = dets[i, -1]
    plt.gca().add_patch(
        plt.Rectangle((bbox[0], bbox[1]), 
                       bbox[2] - bbox[0],
                       bbox[3] - bbox[1], 
                       fill=False, edgecolor='r', linewidth=3)
        )

  if out_dire and len(out_dire) > 0:
    im_name  = im_path.rsplit("/", 1)[1]
    im_name  = im_name.rsplit(".", 1)[0]
    im_path2 = out_dire + im_name + im_ext
    print "\nimage path:", im_path2
    plt.savefig(im_path2)
  else:
    print "\nshow image..."
    plt.show()

  plt.clf() # clear the figure handler

def t_bbox2p_bbox_by_hand(t_bbox, w, h, t_ratio=0.72):
  '''
  Given torso bbox, output corresponding person bbox by hand (ratio), 
  it's suitable for upper body and not open-hand images
  where w and h is the size of input image.
  '''
  # Torso
  x1 = t_bbox[0]
  y1 = t_bbox[1]
  x2 = t_bbox[2]
  y2 = t_bbox[3]
  # Person
  diff_x = x1 - x2
  diff_y = y1 - y2
  t_dit = diff_x * diff_x + diff_y * diff_y
  t_dit = math.sqrt(t_dit)
  t_dit = int(t_dit * t_ratio)

  px1 = x1 - t_dit
  py1 = y1 - t_dit
  px2 = x2 + t_dit
  py2 = y2 + t_dit
  
  px1 = max(1, px1)
  py1 = max(1, py1)
  px2 = min(w - 2, px2)
  py2 = min(h - 2, py2)
  
  p_bbox = [px1, py1, px2, py2]
  return p_bbox