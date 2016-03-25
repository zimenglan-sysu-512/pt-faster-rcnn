# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os, sys
import time
import os.path as osp
import PIL
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
import datasets
import cPickle
from time import sleep

class imdb(object):
  """Image database."""

  def __init__(self, name, data="data", cache="cache"):
    self._name = name
    self._num_classes = 0
    self._classes = []
    self._image_cls = {}
    self._image_index = []
    self._obj_proposer = 'selective_search'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    self.config = {}
    self._data = data
    self._cache = cache

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def image_index(self):
      return self._image_index

  @property
  def image_cls(self):
    return self._image_cls

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    '''cfg.TRAIN.PROPOSAL_METHOD'''
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def cache(self):
    return self._cache

  @property
  def data(self):
    return self._data

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(datasets.ROOT_DIR, \
        self._data, self._cache))
    if not os.path.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  @property
  def o_num_images(self): 
    '''origin number of images whether they are flipped or not'''
    num_images = self.num_images
    if cfg.TRAIN.USE_FLIPPED:
      nmod = num_images % 2
      assert nmod == 0
      o_num_images = num_images / 2
    else:
      o_num_images = num_images
    return o_num_images

  def image_name_at(self, i):
    raise NotImplementedError
  
  def image_path_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def cache_rpn_roidb(self, rpn_file_ext=".pkl"):
    raise NotImplementedError

  # the same as lib/roi_data_layer/roidb.py _compute_targets functiion
  @staticmethod
  def compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0]  = labels[ex_inds]
    targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets

  # ###########################################################
  #   roidb
  # ###########################################################

  @staticmethod
  def widths_heights_sizes_prefix(prefix):
    cfg.TRAIN.SIZES_PREFIX   = prefix
    cfg.TRAIN.WIDTHS_PREFIX  = prefix
    cfg.TRAIN.HEIGHTS_PREFIX = prefix

  def comp_widths(self, num_images):
    print "start getting widths..."

    widths_prefix  = cfg.TRAIN.WIDTHS_PREFIX 
    widths_postfix = cfg.TRAIN.WIDTHS_POSTFIX 
    cache_file = os.path.join(self.cache_path, self._input_img_debug_str \
        + widths_prefix + widths_postfix)

    if os.path.exists(cache_file):
      print "start loading widths from file"
      with open(cache_file, 'rb') as fid:
        widths = cPickle.load(fid)
      print '{} widths of gt roidb loaded from {}'.format(self.name, cache_file)
    else:
      print "start computing widths from images - ori"
      widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                for i in xrange(num_images)]
      # write
      with open(cache_file, 'wb') as fid:
        cPickle.dump(widths, fid, cPickle.HIGHEST_PROTOCOL)
      print 'wrote widths of gt roidb to {}'.format(cache_file)
    # done about widths
    print "get widths done..."

    # set
    cfg.TRAIN.COMP_WIDTHS_PATH = cache_file
    return widths

  def comp_heights(self, num_images):
    print "start getting heights..."

    heights_prefix  = cfg.TRAIN.HEIGHTS_PREFIX 
    heights_postfix = cfg.TRAIN.HEIGHTS_POSTFIX 
    cache_file = os.path.join(self.cache_path, self._input_img_debug_str \
        + heights_prefix + heights_postfix)

    if os.path.exists(cache_file):
      print "start loading heights from file"
      with open(cache_file, 'rb') as fid:
        # mornal loading for pkl
        heights = cPickle.load(fid)
      print '{} heights of gt roidb loaded from {}'.format(self.name, cache_file)
    else:
      print "start computing heights from images - ori"
      heights = [PIL.Image.open(self.image_path_at(i)).size[1]
                for i in xrange(num_images)]
      # write
      with open(cache_file, 'wb') as fid:
        cPickle.dump(heights, fid, cPickle.HIGHEST_PROTOCOL)
      print 'wrote heights of gt roidb to {}'.format(cache_file)
    # done about heights
    print "get heights done..."

    # set 
    cfg.TRAIN.COMP_HEIGHTS_PATH = cache_file
    return heights

  def comp_sizes(self, num_images):
    print "start getting sizes..."
    
    sizes_prefix  = cfg.TRAIN.SIZES_PREFIX 
    sizes_postfix = cfg.TRAIN.SIZES_POSTFIX 
    cache_file = os.path.join(self.cache_path, self._input_img_debug_str \
        + sizes_prefix + sizes_postfix)

    if os.path.exists(cache_file):
      print "start loading sizes from file"
      with open(cache_file, 'rb') as fid:
        # mornal loading for pkl
        sizes = cPickle.load(fid)
      print '{} sizes of gt roidb loaded from {}'.format(self.name, cache_file)
    else:
      print "start computing sizes from images - ori"
      sizes = [PIL.Image.open(self.image_path_at(i)).size
                for i in xrange(num_images)]
      # write
      with open(cache_file, 'wb') as fid:
        cPickle.dump(sizes, fid, cPickle.HIGHEST_PROTOCOL)
      print 'wrote sizes of gt roidb to {}'.format(cache_file)
    # done about sizes
    print "get sizes done..."

    # set 
    cfg.TRAIN.COMP_SIZES_PATH = cache_file
    return sizes

  def comp_widths_heights(self, sizes):
    # widths
    widths_prefix  = cfg.TRAIN.WIDTHS_PREFIX 
    widths_postfix = cfg.TRAIN.WIDTHS_POSTFIX 
    w_cache_file = os.path.join(self.cache_path, self._input_img_debug_str \
        + widths_prefix + widths_postfix)
    # heights
    heights_prefix  = cfg.TRAIN.HEIGHTS_PREFIX 
    heights_postfix = cfg.TRAIN.HEIGHTS_POSTFIX 
    h_cache_file = os.path.join(self.cache_path, self._input_img_debug_str \
        + heights_prefix + heights_postfix)

    # widths
    if os.path.exists(w_cache_file):
      print "start loading widths from file"
      with open(w_cache_file, 'rb') as fid:
        widths = cPickle.load(fid)
      print '{} widths of gt roidb loaded from {}'.format(self.name, w_cache_file)
    else:
      widths  = [size[0] for size in sizes]
      with open(w_cache_file, 'wb') as fid:
        cPickle.dump(widths, fid, cPickle.HIGHEST_PROTOCOL)
      print 'wrote sizes of gt roidb to {}'.format(w_cache_file)

    # heights
    if os.path.exists(h_cache_file):
      print "start loading heights from file"
      with open(h_cache_file, 'rb') as fid:
        heights = cPickle.load(fid)
      print '{} heights of gt roidb loaded from {}'.format(self.name, h_cache_file)
    else:
      heights = [size[1] for size in sizes]
      with open(h_cache_file, 'wb') as fid:
        cPickle.dump(heights, fid, cPickle.HIGHEST_PROTOCOL)
      print 'wrote sizes of gt roidb to {}'.format(h_cache_file)

    print 
    print "(widths, heights) loading done..."
    print 

    # set file path
    cfg.TRAIN.COMP_WIDTHS_PATH  = w_cache_file
    cfg.TRAIN.COMP_HEIGHTS_PATH = h_cache_file
    return widths, heights

  def comp_widths_heights_sizes(self, num_images):
    print "start getting (widths, heights, sizes)..."
    # sizes
    sizes_prefix  = cfg.TRAIN.SIZES_PREFIX 
    sizes_postfix = cfg.TRAIN.SIZES_POSTFIX 
    s_cache_file = os.path.join(self.cache_path, self._input_img_debug_str \
        + sizes_prefix + sizes_postfix)

    if os.path.exists(s_cache_file):
      print "start loading sizes from file"
      with open(s_cache_file, 'rb') as fid:
        sizes = cPickle.load(fid)
      print '{} sizes of gt roidb loaded from {}'.format(self.name, s_cache_file)
    else:
      print "start computing sizes from images - ori"
      sizes   = [PIL.Image.open(self.image_path_at(i)).size 
                for i in xrange(num_images)]
      with open(s_cache_file, 'wb') as fid:
        cPickle.dump(sizes, fid, cPickle.HIGHEST_PROTOCOL)
      print 'wrote sizes of gt roidb to {}'.format(s_cache_file)
    
    # set 
    cfg.TRAIN.COMP_SIZES_PATH = s_cache_file

    # widths and heights
    widths, heights = self.comp_widths_heights(sizes)

    print "(widths, heights, sizes) loading done..."
    print 

    return widths, heights, sizes

  @property
  def roidb(self):
    # boxes
    # gt_overlaps
    # gt_classes
    # flipped
    if self._roidb is not None:
      return self._roidb

    print "loading roidb from roidb_handler..."
    self._roidb = self.roidb_handler()
    return self._roidb

  def append_flipped_images(self):
    ''''''
    num_images = self.num_images
    
    print "start getting widths"
    widths = self.comp_widths(num_images)
    print "num images:", num_images
    print "len of widths:", len(widths)

    # Get roidb, for the reason of multiprocess of python
    print "start getting roidb"
    _ = self.roidb
    print "finish getting roidb"

    print "start flipping images..."
    for i in xrange(num_images):
      if i % cfg.TRAIN.PRINT_ITER_NUN == 0:
        print "im:", i, num_images
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      for j in xrange(len(boxes)):
        if boxes[j, 0] > boxes[j, 2]:
          boxes[j, 0], boxes[j, 2] = boxes[j, 2], boxes[j, 0]
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'boxes' : boxes,
               'gt_overlaps' : self.roidb[i]['gt_overlaps'],
               'gt_classes' : self.roidb[i]['gt_classes'],
               'flipped' : True}
      self.roidb.append(entry)
    print "flipped im:", self.num_images
    print "processing image index and cls in lib/dataset/imdb.py ..."
    print

    # double image index and image cls
    self._image_index = self._image_index * 2
    print "before flipped, num_images:", num_images
    for k in self._image_cls:
      image_cls2 = []
      for  ic in self._image_cls[k]:
        image_cls2.append(ic + num_images)
      self._image_cls[k].extend(image_cls2)

  def create_roidb_from_box_list(self, box_list, gt_roidb):
    print "num_images: %s before create roidb from box list" % (self.num_images,)
    print "num_bboxes: %s before create roidb from box list" % (len(box_list),)
    print "num_gt_roidb: %s before create roidb from box list" % (len(gt_roidb),)
    assert len(box_list) == self.num_images, \
        'Number of boxes must match number of ground-truth images' \
        'len(box_list):%s num_images: %s' % (len(box_list), self.num_images)
    assert len(gt_roidb) == self.num_images, \
        'Number of gt roidb must match number of ground-truth images' \
        'len(box_list):%s num_images: %s' % (len(gt_roidb), self.num_images)

    roidb = []
    for i in xrange(self.num_images):
      if i % cfg.TRAIN.PRINT_ITER_NUN == 0:
        print "bb:", i, self.num_images
      
      boxes     = box_list[i]
      num_boxes = boxes.shape[0]
      overlaps  = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if gt_roidb is not None:
        gt_boxes = gt_roidb[i]['boxes']
        gt_classes = gt_roidb[i]['gt_classes']
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
        
      # set
      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({'boxes' : boxes,
                    'gt_classes' : np.zeros((num_boxes,),
                                            dtype=np.int32),
                    'gt_overlaps' : overlaps,
                    'flipped' : False})
    print "bbox list im:", self.num_images
    print "bbox list done..."
    print "len of roidb:", len(roidb)
    print
    sleep(3)
    return roidb

  @staticmethod
  def merge_roidbs(a, b):
    assert len(a) == len(b)
    num_ab = len(a)
    for i in xrange(num_ab):
      if i % cfg.TRAIN.PRINT_ITER_NUN == 0:
        print "merge roidb: %s-th (%s)" % (i, num_ab,)
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
    return a

  @staticmethod
  def merge_p_roidb(a, b):
    a['boxes']       = np.vstack((a['boxes'], b['boxes']))
    a['gt_classes']  = np.hstack((a['gt_classes'], b['gt_classes']))
    a['gt_overlaps'] = scipy.sparse.vstack([a['gt_overlaps'], b['gt_overlaps']])
    return a

  # ###########################################################
  #   evaluation
  # ###########################################################

  def evaluate_detections(self, all_boxes, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.
    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    raise NotImplementedError

  def evaluate_recall(self, candidate_boxes=None, ar_thresh=0.5):
    # Record max overlap value for each gt box
    # Return vector of overlap values
    gt_overlaps = np.zeros(0)
    for i in xrange(self.num_images):
      gt_inds = np.where(self.roidb[i]['gt_classes'] > 0)[0]
      gt_boxes = self.roidb[i]['boxes'][gt_inds, :]

      if candidate_boxes is None:
        non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
        boxes = self.roidb[i]['boxes'][non_gt_inds, :]
      else:
        boxes = candidate_boxes[i]
      if boxes.shape[0] == 0:
        continue
      overlaps = bbox_overlaps(boxes.astype(np.float),
                               gt_boxes.astype(np.float))

      # gt_overlaps = np.hstack((gt_overlaps, overlaps.max(axis=0)))
      _gt_overlaps = np.zeros((gt_boxes.shape[0]))
      for j in xrange(gt_boxes.shape[0]):
        argmax_overlaps = overlaps.argmax(axis=0)
        max_overlaps = overlaps.max(axis=0)
        gt_ind = max_overlaps.argmax()
        gt_ovr = max_overlaps.max()
        assert(gt_ovr >= 0)
        box_ind = argmax_overlaps[gt_ind]
        _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        assert(_gt_overlaps[j] == gt_ovr)
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1

      gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    num_pos = gt_overlaps.size
    gt_overlaps = np.sort(gt_overlaps)
    step = 0.001
    thresholds = np.minimum(np.arange(0.5, 1.0 + step, step), 1.0)
    recalls = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
      recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    ar = 2 * np.trapz(recalls, thresholds)

    return ar, gt_overlaps, recalls, thresholds

  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass