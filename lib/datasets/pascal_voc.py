# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Dengke Dong
# --------------------------------------------------------

import datasets
import datasets.pascal_voc
import os
import time
import datasets.imdb
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.config import cfg
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
from time import sleep

class pascal_voc(datasets.imdb):
  def __init__(self, image_set, year, \
      devkit_path=None, image_ext=".jpg", \
      D_INPUT_DIR="", D_INPUT_IMG_DIR="", \
      D_INPUT_LAB_DIR="", D_INPUT_FILE="", \
      data="data", cache="cache"):
    # #####################################################################
    datasets.imdb.__init__(self, 'voc_' + year + '_' + image_set, \
        data=data, cache=cache)
    self._year = year
    self._image_set = image_set
    self._devkit_path = self._get_default_path() if devkit_path is None \
                        else devkit_path
    self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)

    self._classes = []
    self._classes_filepath = cfg.CLASSES_FILEPATH
    if self._classes_filepath is not None:
      self._classes_filepath = self._classes_filepath.strip()
    else:
      self._classes_filepath = ""
    cls_filepath = self._classes_filepath
    if len(cls_filepath) > 0 and os.path.exists(cls_filepath):
      print "class names' file path:", cls_filepath
      with open(cls_filepath) as f:
        self._classes = [x.strip().lower() for x in f.readlines()]
      self._classes = tuple(self._classes)
    if len(self._classes) <= 0:
      print "Missing classes_filepath"
      print "Here we use PascalVoc2012 actions classes ..."
      print
      self._classes = ('__background__', # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')

    print "classes_filepath", cls_filepath
    print "Class Names:"
    print self._classes
    print 
    self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
    print
    print "Corresponding class Names' indices:"
    print self._class_to_ind
    print
    # D_IS_INPUT
    self._image_ext = cfg.IMAGE_EXT
    # self._image_ext = image_ext
    self._input_img_debug_str = ""
    self._D_INPUT_DIR = D_INPUT_DIR
    self._D_INPUT_FILE = D_INPUT_FILE
    self._D_INPUT_IMG_DIR = D_INPUT_IMG_DIR
    self._D_INPUT_LAB_DIR = D_INPUT_LAB_DIR
    print "D_INPUT_DIR:", self._D_INPUT_DIR
    print "D_INPUT_FILE:", self._D_INPUT_FILE
    print "D_INPUT_IMG_DIR:", self._D_INPUT_IMG_DIR
    print "D_INPUT_LAB_DIR:", self._D_INPUT_LAB_DIR
    self._D_IS_INPUT = self._check_d_detection()
    self._image_index = self._load_image_set_index()
    self._input_img_debug_num = int(cfg.INPUT_IMG_DEBUG_NUM)
    # Debug
    if self._input_img_debug_num > 0:
      self._input_img_debug_num = min(len(self._image_index), \
          self._input_img_debug_num)
      self._image_index = self._image_index[: self._input_img_debug_num]
      self._input_img_debug_str = \
          cfg.INPUT_DEBUG_STR + str(self._input_img_debug_num)

    # Default (roidb handler)
    self._roidb_handler = self.selective_search_roidb
    
    # PASCAL specific config options
    self.config = {'cleanup'  : True,
                   'use_salt' : True,
                   'top_k'    : 2000,
                   'use_diff' : False,
                   'rpn_file' : None,   # file or cache directory
                   'roidbs_cache_path:': None}
    assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
    assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)
    print   
    print "_D_INPUT_DIR:", self._D_INPUT_DIR
    print "_D_INPUT_FILE:", self._D_INPUT_FILE
    print "_D_INPUT_IMG_DIR:", self._D_INPUT_IMG_DIR
    print "_D_INPUT_LAB_DIR:", self._D_INPUT_LAB_DIR
    print "_D_IS_INPUT:", self._D_IS_INPUT

    print
    print "The number of input images:", len(self._image_index)
    print "Finish initializing pascal_voc instance..."
    print "Start creating roidb in lib/datasets/pascal_voc.py..."
    print
    time.sleep(3)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_name_at(self, i):
    """
    Return the name to image i in the image sequence.
    """
    return self._image_index[i]

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    if self._D_IS_INPUT:
      image_path = self._D_INPUT_DIR + self._D_INPUT_IMG_DIR + \
          index + self._image_ext
    else:
      image_path = os.path.join(self._data_path, 'JPEGImages',
                                index + self._image_ext)
    assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)

    return image_path

  def _check_d_detection(self):
    '''
    Different input format from Pascal Voc
    '''
    if self._D_INPUT_DIR is None or len(self._D_INPUT_DIR) <= 0:
      return False
    if self._D_INPUT_FILE is None or len(self._D_INPUT_FILE) <= 0:
      return False
    if self._D_INPUT_IMG_DIR is None or len(self._D_INPUT_IMG_DIR) <= 0:
      return False
    if self._D_INPUT_LAB_DIR is None or len(self._D_INPUT_LAB_DIR) <= 0:
      return False
    return True

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    if self._D_IS_INPUT:
      print
      print "Load dataset from ", self._D_INPUT_DIR
      print "In lib/datasets/pascal_voc.py _load_image_set_index function..."
      print 
      image_set_file = self._D_INPUT_DIR + self._D_INPUT_LAB_DIR \
          + self._D_INPUT_FILE
      assert os.path.exists(image_set_file), \
              'Path does not exist: {}'.format(image_set_file)
      # imgidx objidx1, bbox1, cls1, objidx2, bbox2, cls2, ...
      #   where bbox is four-tuple (x1, y1, x2, y2)
      with open(image_set_file) as f:
        k_id = 0
        image_index = []
        for x in f.readlines():
          x = x.strip().split()
          vlen = len(x)
          if (vlen - 1) % 6 != 0:
            err_str = 'Input format must be: imgidx objidx1 ' \
                + 'x1 y1 x2 y2 cls1 objidx2 x1 y1 x2 y2 cls2 ...'
            raise IOError(err_str)
          num_objs = (vlen - 1) / 6
          imgidx, info = x[0], x[1:]

          for idx in range(0, num_objs):
            idx2 = idx * 6
            cls = info[idx2 + 5].strip().lower()
            cls = self._class_to_ind[cls]
            if cls not in self._image_cls.keys():
              self._image_cls[cls] = []
            self._image_cls[cls].append(k_id)
            
          k_id += 1
          image_index.append(imgidx.strip())
      k_keys = self._image_cls.keys()
      k_keys.sort()
      for k in k_keys:
        print "cls ind:", k, " cls num:", len(self._image_cls[k])
        print "max idx:", max(self._image_cls[k]), "min idx:", min(self._image_cls[k])
      print 
      time.sleep(3)
    else :
      print 
      print "Load datasets from ", self._data_path
      print "In lib/datasets/pascal_voc.py _load_image_set_index function..."
      print 
      # Example path to image set file:
      # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
      image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                    self._image_set + '.txt')
      assert os.path.exists(image_set_file), \
              'Path does not exist: {}'.format(image_set_file)
      with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]

    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(datasets.ROOT_DIR, 'data', 'VOCdevkit' + self._year)


  # ##################################################################
  #   Use For Faster RCNN
  # ##################################################################
  
  # Load gt roidb for rpn training
  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    # self.name: 'voc_' + year + '_' + image_set
    roidb_postfix = '_gt_roidb.pkl'
    cache_file = os.path.join(self.cache_path, self.name + \
        self._input_img_debug_str + roidb_postfix)

    if os.path.exists(cache_file):
      print 'start loading {} gt roidb from {}'.format(self.name, cache_file)
      with open(cache_file, 'rb') as fid:
        roidb = cPickle.load(fid)
      print 'finish loading {} gt roidb from {}'.format(self.name, cache_file)
      return roidb

    # if not exist, create
    gt_roidb = self._load_pascal_annotation()
    # write into file
    print 'start writing gt roidb to {}'.format(cache_file)
    with open(cache_file, 'wb') as fid:
      cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'finish writing gt roidb to `{}` done'.format(cache_file)

    return gt_roidb

  # Load RPN Proposals for fast rcnn training
  # 1 load gt roidb
  # 2 load rpn props (box_list)
  # 3 create_roidb_from_box_list
  # 4 merge gt roidb and box_list
  # 5 flip image
  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      print "start merging gt_roidb with rpn_roidb"
      roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
      print "finishing merging gt_roidb with rpn_roidb"
    else:
      roidb = self._load_rpn_roidb(None)

    print "return roidb in lib/dataset/pascal_voc.py -- rpn_roidb func"
    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    ''''''
    filename = self.config['rpn_file']
    print "filename:", filename

    # filename2 = filename.rsplit(".", 1)[0] + ".memmap"
    # print "filename2:", filename2

    print 'Loading proposals from {}'.format(filename)
    assert os.path.exists(filename), \
           'rpn data not found at: {}'.format(filename)

    ## normal loading for pkl file
    # with open(filename, 'rb') as f:
    #   box_list = cPickle.load(f)

    ## loading by np.memmap for memory reduction
    box_list = np.load(filename, mmap_mode='r')

    print "in lib/dataset/pascal_voc.py -- _load_rpn_roidb func..."
    if self._input_img_debug_num > 0:
      print "before num of roidb:", len(box_list)
      box_list = box_list[: self._input_img_debug_num]
      print "after num of roidb:", len(box_list)
    else:
      print "use origin proposals without debug num..."

    print "Loading proposals done..."
    
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  # Load rpn proposals for fast rcnn training
  # But, per image per rpn proposal file
  def cache_rpn_roidb(self, pkl_file_ext=cfg.PKL_FILE_EXT):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.
    This function loads/saves from/to a cache files (per image per file) 
      to speed up future calls.
    """
    num_images = self.num_images
    sizes = self.comp_sizes(num_images)
    _, _, = self.comp_widths_heights(sizes)
    print "num sizes:",  len(sizes)
    print "num images:", num_images
    assert num_images == len(sizes)
    print "loading sizes of input images done..."
    sleep(3)

    # ground-truths bboxes and class labels
    print "start loading gt_roidb"
    gt_roidb = self.gt_roidb()
    assert num_images == len(gt_roidb), 'does match num_images with len of gt_roidb'
    print "loading gt_roidb done..."
    sleep(3)

    # cfg.TRAIN.RPN_CACHE_PATH
    rpn_cache_path    = self.config['rpn_file'] 
    # cfg.TRAIN.ROIDBS_CACHE_PATH
    roidbs_cache_path = self.config['roidbs_cache_path']  

    for idx in xrange(num_images):
      if idx % cfg.TRAIN.PRINT_ITER_NUN == 0:
        print "cache -> im:", idx, num_images

      # ##########################################################################
      # gt bounding boxes
      p_gt_roidb = gt_roidb[idx]  # per image

      # ##########################################################################
      # rpn proposals
      im_name = self.image_name_at(idx) 
      im_path = self.image_path_at(idx)
      # rpn cache file of rpn proposals generated by the rpn network
      rpn_cache_file = rpn_cache_path + im_name + pkl_file_ext  
      # load from file
      assert os.path.exists(rpn_cache_file), \
          'rpn data not found at: {}'.format(rpn_cache_file)
      with open(rpn_cache_file, 'rb') as fid:
        rpn_dict = cPickle.load(fid)
      rpn_boxes = rpn_dict["boxes"]
      imgidx    = rpn_dict["imgidx"]
      assert imgidx == im_name, 'does match imgidx to im_name'
      num_boxes = rpn_boxes.shape[0]

      # ##########################################################################
      # overlaps
      overlaps  = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if p_gt_roidb is not None:
        gt_boxes    = p_gt_roidb['boxes']
        gt_classes  = p_gt_roidb['gt_classes']
        # calculate the overlap
        gt_overlaps = bbox_overlaps(rpn_boxes.astype(np.float), gt_boxes.astype(np.float))
        argmaxes    = gt_overlaps.argmax(axis=1)
        maxes       = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
      overlaps = scipy.sparse.csr_matrix(overlaps)
       
      # ##########################################################################
      # p_rpn_roidb
      p_rpn_roidb = {'boxes':       rpn_boxes,
                     'gt_classes':  np.zeros((num_boxes,), dtype=np.int32),
                     'gt_overlaps': overlaps,
                     'flipped':     False}

      # ##########################################################################
      # p_roidb -- by merge
      p_roidb = datasets.imdb.merge_p_roidb(p_gt_roidb, p_rpn_roidb)
      p_roidb['image']  = im_path
      p_roidb['width']  = sizes[idx][0]
      p_roidb['height'] = sizes[idx][1]

      # need gt_overlaps as a dense array for argmax
      gt_overlaps = p_roidb['gt_overlaps'].toarray()
      
      # max overlap with gt over classes (columns)
      max_overlaps = gt_overlaps.max(axis=1)
      
      # gt class that had the max overlap
      max_classes = gt_overlaps.argmax(axis=1)
      p_roidb['max_classes']  = max_classes
      p_roidb['max_overlaps'] = max_overlaps
      
      # sanity checks
      # max overlap of 0 => class should be zero (background)
      zero_inds = np.where(max_overlaps == 0)[0]
      assert all(max_classes[zero_inds] == 0)
      
      # max overlap > 0 => class should not be zero (must be a fg class)
      nonzero_inds = np.where(max_overlaps > 0)[0]
      assert all(max_classes[nonzero_inds] != 0)

      # ##########################################################################
      # flip to get p_roidb2
      if cfg.TRAIN.USE_FLIPPED:
        boxes = p_roidb['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = sizes[idx][0] - oldx2 - 1
        boxes[:, 2] = sizes[idx][0] - oldx1 - 1
        for j in xrange(len(boxes)):
          if boxes[j, 0] > boxes[j, 2]:
            boxes[j, 0], boxes[j, 2] = boxes[j, 2], boxes[j, 0]
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        p_roidb2 = {'boxes':       boxes,
                    'gt_overlaps': p_roidb['gt_overlaps'],
                    'gt_classes' : p_roidb['gt_classes'],
                    'flipped':     True}
        p_roidb2['image']        = im_path
        p_roidb2['width']        = sizes[idx][0]
        p_roidb2['height']       = sizes[idx][1]
        p_roidb2['max_classes']  = max_classes
        p_roidb2['max_overlaps'] = max_overlaps

      # ##########################################################################
      # print "write per roidb into file"
      p_roidb_cache_file  = roidbs_cache_path + im_name + pkl_file_ext
      with open(p_roidb_cache_file, 'wb') as fid:
        cPickle.dump(p_roidb, fid, cPickle.HIGHEST_PROTOCOL)
      
      if cfg.TRAIN.USE_FLIPPED:
        p_roidb_cache_file2 = roidbs_cache_path + im_name + \
            cfg.FLIPPED_POSTFIX + pkl_file_ext
        with open(p_roidb_cache_file2, 'wb') as fid:
          cPickle.dump(p_roidb2, fid, cPickle.HIGHEST_PROTOCOL)

    print "preprocess roidb done..."
    print 
    sleep(3)
    
    # ##########################################################################
    # double image index and image cls
    if cfg.TRAIN.USE_FLIPPED:
      print "before flipped, num_images:", num_images
      self._image_index = self._image_index * 2
      for k in self._image_cls:
        image_cls2 = []
        for  ic in self._image_cls[k]:
          image_cls2.append(ic + num_images)
        self._image_cls[k].extend(image_cls2)

    if cfg.TRAIN.BBOX_REG:
      print "start bbox reg computing in cache_rpn_roidb func of pascal_voc.py"
      cfg.TRAIN.BBOX_REG_NORMALIZE_MEANS, cfg.TRAIN.BBOX_REG_NORMALIZE_STDS = \
          self.add_bbox_regression_targets()
      print "finish bbox reg computing in cache_rpn_roidb func of pascal_voc.py"

    # roidb is set to be None
    return None

  # modification of lib/roi_data_layer/roidb.py add_bbox_regression_targets functiion
  def add_bbox_regression_targets(self, pkl_file_ext=cfg.PKL_FILE_EXT):
    """Add information needed to train bounding-box regressors."""
    # Infer number of classes from the number of columns in gt_overlaps
    num_images        = self.num_images
    num_classes       = self.num_classes
    o_num_images      = self.o_num_images 
    roidbs_cache_path = self.config['roidbs_cache_path']  

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      # Use fixed / precomputed "means" and "stds" instead of empirical values
      stds  = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS),  (num_classes, 1))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
    else:
      # Compute values needed for means and stds
      # var(x) = E(x^2) - E(x)^2
      sums = np.zeros((num_classes, 4))
      squared_sums = np.zeros((num_classes, 4))
      class_counts = np.zeros((num_classes, 1)) + cfg.EPS

      for im_i in xrange(num_images):
        if im_i % cfg.TRAIN.PRINT_ITER_NUN == 0:
          print "bb reg - im_i: %s (%s)" % (im_i, num_images)
        # read
        p_roidb_cache_file = None
        im_name = self.image_name_at(im_i)
        if im_i < o_num_images: # not flipped
          p_roidb_cache_file = roidbs_cache_path + im_name + pkl_file_ext
        else:                   # flipped
          p_roidb_cache_file = roidbs_cache_path + im_name + \
              cfg.FLIPPED_POSTFIX + pkl_file_ext
        with open(p_roidb_cache_file, 'rb') as fid:
          p_roidb = cPickle.load(fid)
        # compute bbox_targets
        rois         = p_roidb['boxes']
        max_overlaps = p_roidb['max_overlaps']
        max_classes  = p_roidb['max_classes']
        p_roidb['bbox_targets'] = datasets.imdb.compute_targets(rois, max_overlaps, max_classes)
        # write
        with open(p_roidb_cache_file, 'wb') as fid:
          cPickle.dump(p_roidb, fid, cPickle.HIGHEST_PROTOCOL)

        targets = p_roidb['bbox_targets']
        for cls in xrange(1, num_classes):
          cls_inds = np.where(targets[:, 0] == cls)[0]
          if cls_inds.size > 0:
            class_counts[cls]    += cls_inds.size
            sums[cls, :]         += targets[cls_inds, 1:].sum(axis=0)
            squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)
      print "bb reg - im_i: %s (%s)" % (num_images, num_images)
      means = sums / class_counts
      stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0) # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0) # ignore bg class
    print 

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
      print "Normalizing targets"
      for im_i in xrange(num_images):
        if im_i % cfg.TRAIN.PRINT_ITER_NUN == 0:
          print "norm bb reg - im_i: %s (%s)" % (im_i, num_images)
        # read
        p_roidb_cache_file = None
        im_name = self.image_name_at(im_i)
        if im_i < o_num_images: # not flipped
          p_roidb_cache_file = roidbs_cache_path + im_name + pkl_file_ext
        else:                   # flipped
          p_roidb_cache_file = roidbs_cache_path + im_name + \
              cfg.FLIPPED_POSTFIX + pkl_file_ext
        with open(p_roidb_cache_file, 'rb') as fid:
          p_roidb = cPickle.load(fid)

        targets = p_roidb['bbox_targets']
        for cls in xrange(1, num_classes):
          cls_inds = np.where(targets[:, 0] == cls)[0]
          p_roidb['bbox_targets'][cls_inds, 1:] -= means[cls, :]
          p_roidb['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
        # write
        with open(p_roidb_cache_file, 'wb') as fid:
          cPickle.dump(p_roidb, fid, cPickle.HIGHEST_PROTOCOL)
      print "norm bb reg - im_i: %s (%s)" % (num_images, num_images)
    else:
      print "NOT normalizing targets"

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()
 
  # ##################################################################
  #   Use For Fast RCNN
  # ##################################################################

  def selective_search_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    roidb_postfix = '_selective_search_roidb.pkl'
    cache_file = os.path.join(self.cache_path, \
       self.name + self._input_img_debug_str + roidb_postfix)

    print "cache file:", cache_file
    print "in lib/dataset/pascal_voc.py -- selective_search_roidb func."

    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = cPickle.load(fid)
      print '{} ss roidb loaded from {}'.format(self.name, cache_file)
      return roidb

    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      ss_roidb = self._load_selective_search_roidb(gt_roidb)
      roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
    else:
      roidb = self._load_selective_search_roidb(None)
    with open(cache_file, 'wb') as fid:
      cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote ss roidb to {}'.format(cache_file)

    return roidb

  def _load_selective_search_roidb(self, gt_roidb):
    filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                            'selective_search_data',
                                            self.name + '.mat'))
    assert os.path.exists(filename), \
           'Selective search data not found at: {}'.format(filename)
    raw_data = sio.loadmat(filename)['boxes'].ravel()

    box_list = []
    for i in xrange(raw_data.shape[0]):
      box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_pascal_annotation_pas(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    def get_data_from_tag(node, tag):
      return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
      data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    if not self.config['use_diff']:
      # Exclude the samples labeled as difficult
      non_diff_objs = [obj for obj in objs
                       if int(get_data_from_tag(obj, 'difficult')) == 0]
      if len(non_diff_objs) != len(objs):
        print 'Removed {} difficult objects' \
            .format(len(objs) - len(non_diff_objs))
      objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      # Make pixel indexes 0-based
      x1 = float(get_data_from_tag(obj, 'xmin')) - 1
      y1 = float(get_data_from_tag(obj, 'ymin')) - 1
      x2 = float(get_data_from_tag(obj, 'xmax')) - 1
      y2 = float(get_data_from_tag(obj, 'ymax')) - 1
      cls = str(get_data_from_tag(obj, "name")).lower().strip()
      cls = self._class_to_ind[cls]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes' : boxes,
            'gt_classes': gt_classes,
            'gt_overlaps' : overlaps,
            'flipped' : False}

  def _load_pascal_annotation_others(self):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC format.
    """
    filename = self._D_INPUT_DIR + self._D_INPUT_LAB_DIR \
        + self._D_INPUT_FILE
    assert os.path.exists(filename), \
            'Path does not exist: {}'.format(filename)
    # imgidx objidx1, bbox1, cls1, objidx2, bbox2, cls2, ...
    #   where bbox is four-tuple (x1, y1, x2, y2)
    k_id = 0
    gt_roidb = []
    image_index2 = []
    with open(filename) as f:
      for line in f.readlines():
        line = line.strip()
        info = line.split()
        vlen = len(info)
        if (vlen - 1) % 6 != 0:
          err_str = 'Input format must be: imgidx objidx1 ' \
              + 'x1 y1 x2 y2 cls1 objidx2 x1 y1 x2 y2 cls2 ...'
          raise IOError(err_str)
        if (vlen - 1) / 6 <= 0:
          err_str = 'Input format must be: imgidx objidx1 ' \
              + 'x1 y1 x2 y2 cls1 objidx2 x1 y1 x2 y2 cls2 ...'
          raise IOError(line)
        num_objs = (vlen - 1) / 6
        if num_objs <= 0:
          print "line:", line
        assert num_objs > 0, 'invalid input: %s' % (line,)
        objs = []
        imgidx, info = info[0], info[1:]
        image_index2.append(imgidx)

        for idx in range(0, num_objs):
          idx2 = idx * 6
          objidx = int(info[idx2 + 0].strip())
          x1   = float(info[idx2 + 1].strip())
          y1   = float(info[idx2 + 2].strip())
          x2   = float(info[idx2 + 3].strip())
          y2   = float(info[idx2 + 4].strip())
          cls   =      info[idx2 + 5].strip().lower()
          if x1 > x2:
            x1, x2 = x2, x1
          if y1 > y2:
            y1, y2 = y2, y1
          ltuple = (objidx, x1, y1, x2, y2, cls)
          objs.append(ltuple)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # make sure that x1 y1 x2 y2 are valid 
        for idx in range(0, num_objs):
          x1 = max(objs[idx][1] - 1, 2)
          y1 = max(objs[idx][2] - 1, 2)
          x2 = objs[idx][3] - 2
          y2 = objs[idx][4] - 2
          x1 = max(2, x1)
          y1 = max(2, y1)
          cls = self._class_to_ind[objs[idx][5].lower().strip()]
          boxes[idx, :] = [x1, y1, x2, y2]
          gt_classes[idx] = cls
          overlaps[idx, cls] = 1.0
        # self-increase
        k_id += 1
        if k_id % cfg.TRAIN.PRINT_ITER_NUN == 0:
          print "k_id:", k_id, " num_objs: ", num_objs

        overlaps = scipy.sparse.csr_matrix(overlaps)
        a_roidb = {
                 'boxes' : boxes,
                 'gt_classes': gt_classes,
                 'gt_overlaps' : overlaps,
                 'flipped' : False}
        gt_roidb.append(a_roidb)
    print 
    assert k_id == len(gt_roidb)
    
    if self._input_img_debug_num > 0:
      gt_roidb = gt_roidb[: self._input_img_debug_num]
      image_index2 = image_index2[: self._input_img_debug_num]

      cls_inds = xrange(len(image_index2))
      k_keys = self._image_cls.keys()
      k_keys.sort()
      for k in k_keys:
        self._image_cls[k] = list(set(cls_inds) & set(self._image_cls[k]))
    print "k_id:", k_id, " num_objs: ", num_objs, "len(gt_roidb):", len(gt_roidb)

    assert len(gt_roidb) == len(self.image_index)
    assert len(self.image_index) ==len(image_index2)
    for k_id2 in xrange(len(image_index2)):
      assert self.image_index[k_id2] == image_index2[k_id2]

    k_keys = self._image_cls.keys()
    k_keys.sort()
    for k in k_keys:
      print "cls ind:", k, " cls num:", len(self._image_cls[k])
      print "max idx:", max(self._image_cls[k]), \
          " min idx:", min(self._image_cls[k])
    print 
    time.sleep(3)
    
    return gt_roidb

  def _load_pascal_annotation(self):
    gt_roidb = None
    if self._D_IS_INPUT:
      gt_roidb = self._load_pascal_annotation_others()
    else :
      gt_roidb = [self._load_pascal_annotation_pas(index) \
          for index in self.image_index]
    print 
    print "load_pascal_annotation has done..."
    print 
    return gt_roidb

  # ##################################################################
  #   Evaluation
  # ##################################################################

  def _write_voc_results_file(self, all_boxes):
    use_salt = self.config['use_salt']
    comp_id = 'comp4'
    if use_salt:
      comp_id += '-{}'.format(os.getpid())

    # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
    path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                        'Main', comp_id + '_')
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print 'Writing {} VOC results file'.format(cls)
      filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
             continue
          # the VOCdevkit expects 1-based indices
          for k in xrange(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index, dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))
    return comp_id

  def _do_matlab_eval(self, comp_id, output_dir='output'):
    rm_results = self.config['cleanup']

    path = os.path.join(os.path.dirname(__file__),
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
           .format(self._devkit_path, comp_id,
                   self._image_set, output_dir, int(rm_results))
    print('Running:\n{}'.format(cmd))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    comp_id = self._write_voc_results_file(all_boxes)
    self._do_matlab_eval(comp_id, output_dir)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

if __name__ == '__main__':
  d = datasets.pascal_voc('trainval', '2007')
  res = d.roidb
  from IPython import embed; embed()
