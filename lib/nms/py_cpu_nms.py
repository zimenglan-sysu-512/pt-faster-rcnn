# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

# iou: True - area(interset(A, B)) / area(B)
#      False - area(interset(A, B)) / area(union(A, B))
# is_merge:
def py_cpu_nms(dets, thresh, is_merge=False, iou=False):
  """Pure Python NMS baseline."""
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]
  # area of each bbox
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  order = scores.argsort()[::-1]
  # result
  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])
    # interset with the rest by the order
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    if iou:
      ovr = inter / (areas[order[1:]])
    else:
      ovr = inter / (areas[i] + areas[order[1:]] - inter)
    # merge
    if is_merge:
      inds2 = np.where(ovr > thresh)
      merge_set = order[inds2 + 1]
      for ms in merge_set:
        dets[i, 0] = min(dets[i, 0], x1[ms])
        dets[i, 1] = max(dets[i, 1], y1[ms])
        dets[i, 2] = min(dets[i, 2], x2[ms])
        dets[i, 3] = max(dets[i, 3], y2[ms])
    # discard the interset {B}
    inds = np.where(ovr <= thresh)[0]
    order = order[inds + 1]

  return keep
