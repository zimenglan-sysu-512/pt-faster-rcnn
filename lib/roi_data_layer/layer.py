# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import os
import sys
import yaml
import caffe
import cPickle
import numpy as np
from time import sleep
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
from multiprocessing import Process, Queue

class RoIDataLayer(caffe.Layer):
  """Fast R-CNN data layer used for training."""

  def num_images(self):
    return len(self._image_index)

  def o_num_images(self): 
    '''origin number of images whether they are flipped or not'''
    num_images   = self.num_images()
    o_num_images = num_images
    if cfg.TRAIN.USE_FLIPPED:
      assert num_images % 2 == 0
      o_num_images = o_num_images / 2
    return o_num_images

  def _get_data_bias2balance_perm(self):
    ''''''
    if self._image_cls is None or len(self._image_cls) <= 0:
      return None
    perm = []
    for k in self._image_cls.keys():
      # double shuffle
      np.random.shuffle(self._image_cls[k])
      np.random.shuffle(self._image_cls[k])
      ext_len = min(self._data_bias_num, len(self._image_cls[k]))
      perm.extend(self._image_cls[k][:ext_len])

    perm = list(set(perm))
    print "num of perm:", len(perm)
    print "in lib/roi_data_layer/layer.py"
    print "get_data_bias2balance_perm func."
    return perm

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    if cfg.TRAIN.ASPECT_GROUPING:
      if self._cache_flag:
        widths, heights = None, None
        with open(cfg.TRAIN.COMP_WIDTHS_PATH,  'rb') as fid:
          widths  = cPickle.load(fid)
        with open(cfg.TRAIN.COMP_HEIGHTS_PATH, 'rb') as fid:
          heights = cPickle.load(fid)
        if cfg.TRAIN.USE_FLIPPED:
          widths  = widths  * 2
          heights = heights * 2
        print "widths and heights from cache files"
        assert len(widths) == len(heights)
        widths  = np.array(widths)
        heights = np.array(heights)
      else:
        widths  = np.array([r['width']  for r in self._roidb])
        heights = np.array([r['height'] for r in self._roidb])
        print "widths and heights from roidb"

      print "num of widths:",  len(widths)
      print "num of heights:", len(heights)

      horz       = (widths >= heights)
      vert       = np.logical_not(horz)
      horz_inds  = np.where(horz)[0]
      vert_inds  = np.where(vert)[0]
      inds = np.hstack((
          np.random.permutation(horz_inds),
          np.random.permutation(vert_inds)))
      print "num of inds:", len(inds)
      inds       = np.reshape(inds, (-1, 2))
      row_perm   = np.random.permutation(np.arange(inds.shape[0]))
      inds       = np.reshape(inds[row_perm, :], (-1,))
      self._perm = inds
    else:
      if self._cache_flag:
        self._perm = np.random.permutation(np.arange(len(self.num_images())))
      else:
        self._perm = np.random.permutation(np.arange(len(self._roidb)))

    perm = None
    if self._data_bias_num > 0 and self._image_cls is not None:
      perm = self._get_data_bias2balance_perm()

    if perm is not None and len(perm) > 0:
      self._perm = list(set(self._perm) & set(perm))
      print "len(perm):", len(perm)
      print "len(self._perm):", len(self._perm)
      print "use data bias method to balance the dataset."

    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._perm):
      self._shuffle_roidb_inds()

    db_inds   = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur = self._cur + cfg.TRAIN.IMS_PER_BATCH
    return db_inds

  def _get_next_minibatch(self):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    if cfg.TRAIN.USE_PREFETCH:
      return self._blob_queue.get()
    else:
      db_inds = self._get_next_minibatch_inds()
      if self._cache_flag:
        minibatch_db = []
        pkl_file_ext = cfg.PKL_FILE_EXT
        o_num_images = self.o_num_images()
        for i in db_inds:
          im_name = self._image_index[i]
          if i >= o_num_images:
            cache_file = cfg.TRAIN.ROIDBS_CACHE_PATH + im_name + \
                cfg.FLIPPED_POSTFIX + pkl_file_ext  
          else:
            cache_file = cfg.TRAIN.ROIDBS_CACHE_PATH + im_name + \
                pkl_file_ext
          assert os.path.exists(cache_file), \
              'rpn data not found at: {}'.format(cache_file)
          with open(cache_file, 'rb') as fid:
            p_db = cPickle.load(fid)
          minibatch_db.append(p_db)
          # for mdb in minibatch_db:
          #   print mdb
          #   print
          # print "batch size:", len(minibatch_db)
      else:
        minibatch_db = [self._roidb[i] for i in db_inds]
      # return
      return get_minibatch(minibatch_db, self._num_classes)

  def set_image_cls(self, image_cls):
    self._image_cls     = image_cls
    self._data_bias_num = cfg.TRAIN.DATA_BAIS_NUM
    print "data_bias_num:", self._data_bias_num
    sleep(3)

  def set_image_index(self, image_index):
    self._image_index = image_index

  def set_roidb(self, roidb):
    """Set the roidb to be used by this layer during training."""
    # set roidb & cache flag
    self._roidb      = roidb
    self._cache_flag = False
    if self._roidb is None or len(self._roidb) <= 0:
      self._cache_flag = True
    
    if self._cache_flag:
      print "num of roidb (from image index):", len(self._image_index)
    else:
      print "num of roidb (from roidb):", len(roidb)
    print "in lib/roi_data_layer/layer.py - set_roidb func"
    sleep(3)

    # shuffle
    self._shuffle_roidb_inds()

    if cfg.TRAIN.USE_PREFETCH:
      self._blob_queue = Queue(10)
      self._prefetch_process = BlobFetcher(self._blob_queue,
                                           self._roidb,
                                           self._image_cls,
                                           self._num_classes)
      self._prefetch_process.start()
      # Terminate the child process when the parent exists
      def cleanup():
        print 'Terminating BlobFetcher'
        self._prefetch_process.terminate()
        self._prefetch_process.join()
      import atexit
      atexit.register(cleanup)

  def setup(self, bottom, top):
    """Setup the RoIDataLayer."""
    # parse the layer parameter string, which must be valid YAML
    layer_params = yaml.load(self.param_str_)

    self._num_classes = layer_params['num_classes']

    self._name_to_top_map = {}

    # data blob: holds a batch of N images, each with 3 channels
    idx = 0
    top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
        max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
    self._name_to_top_map['data'] = idx
    idx += 1

    self._p_disp_in_shape_c = 0
    self._p_disp_in_shape = cfg.TRAIN.P_DISP_IN_SHAPE

    if cfg.TRAIN.HAS_RPN:
      top[idx].reshape(1, 3)
      self._name_to_top_map['im_info'] = idx
      idx += 1

      top[idx].reshape(1, 4)
      self._name_to_top_map['gt_boxes'] = idx
      idx += 1
    else: # not using RPN
      # rois blob: holds R regions of interest, each is a 5-tuple
      # (n, x1, y1, x2, y2) specifying an image batch index n and a
      # rectangle (x1, y1, x2, y2)
      top[idx].reshape(1, 5)
      self._name_to_top_map['rois'] = idx
      idx += 1

      # labels blob: R categorical labels in [0, ..., K] for K foreground
      # classes plus background
      top[idx].reshape(1)
      self._name_to_top_map['labels'] = idx
      idx += 1

      if cfg.TRAIN.BBOX_REG:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        top[idx].reshape(1, self._num_classes * 4)
        self._name_to_top_map['bbox_targets'] = idx
        idx += 1

        # bbox_inside_weights blob: At most 4 targets per roi are active;
        # thisbinary vector sepcifies the subset of active targets
        top[idx].reshape(1, self._num_classes * 4)
        self._name_to_top_map['bbox_inside_weights'] = idx
        idx += 1

        top[idx].reshape(1, self._num_classes * 4)
        self._name_to_top_map['bbox_outside_weights'] = idx
        idx += 1

    print 'RoiDataLayer: name_to_top:', self._name_to_top_map
    assert len(top) == len(self._name_to_top_map)

  def forward(self, bottom, top):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch()

    for blob_name, blob in blobs.iteritems():
      top_ind = self._name_to_top_map[blob_name]
      # Reshape net's input blobs
      top[top_ind].reshape(*(blob.shape))
      # Copy data into net's input blobs
      top[top_ind].data[...] = blob.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

  def reshape(self, bottom, top):
    """Reshaping happens during the call to forward."""
    pass

class BlobFetcher(Process):
  """Experimental class for prefetching blobs in a separate process."""
  def __init__(self, queue, roidb, image_index, image_cls, num_classes):
    super(BlobFetcher, self).__init__()
    self._cur = 0
    self._perm = None
    self._queue = queue
    self._roidb = roidb
    self._image_cls = image_cls
    self._image_index = image_index
    self._num_classes = num_classes
    self._data_bias_num = cfg.TRAIN.DATA_BAIS_NUM
    self._cache_flag = self._roidb is None or len(self._roidb) <= 0

    if self._cache_flag:
      print "num of roidb:", len(self._image_index)
    else:
      print "num of roidb:", len(roidb)
    print "in lib/roi_data_layer/layer.py - BlobFetcher class"
    sleep(3)
    # shuffle
    self._shuffle_roidb_inds()
    # fix the random seed for reproducibility
    np.random.seed(cfg.RNG_SEED)

  def num_images(self):
    return len(self._image_index)

  def o_num_images(self): 
    '''origin number of images whether they are flipped or not'''
    num_images = self.num_images()
    o_num_images = num_images
    if cfg.TRAIN.USE_FLIPPED:
      assert num_images % 2 == 0
      o_num_images = o_num_images / 2
    return o_num_images

  def _get_data_bias2balance_perm(self):
    ''''''
    if self._image_cls is None or len(self._image_cls) <= 0:
      return None
    perm = []
    for k in self._image_cls.keys():
      np.random.shuffle(self._image_cls[k])
      ext_len = min(self._data_bias_num, len(self._image_cls[k]))
      perm.extend(self._image_cls[k][:ext_len])

    perm = list(set(perm))
    return perm

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # TODO(rbg): remove duplicated code
    if self._data_bias_num > 0 and self._image_cls is not None:
      self._perm = self._get_data_bias2balance_perm()
    else:
      self._perm = np.random.permutation(np.arange(len(self._roidb)))
    self._cur = 0

  def _get_next_minibatch_inds(self):
    """Return the roidb indices for the next minibatch."""
    # TODO(rbg): remove duplicated code
    if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._perm):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
    self._cur += cfg.TRAIN.IMS_PER_BATCH
    return db_inds

  def run(self):
    print 'BlobFetcher started'
    while True:
      # get index
      db_inds = self._get_next_minibatch_inds()
      # get minibatch_db
      if self._cache_flag:
        minibatch_db = []
        pkl_file_ext = cfg.PKL_FILE_EXT
        o_num_images = self.o_num_images()
        for i in db_inds:
          im_name = self._image_index[i]
          if i >= o_num_images:
            cache_file = cfg.TRAIN.ROIDBS_CACHE_PATH + im_name + \
                cfg.FLIPPED_POSTFIX + pkl_file_ext  
          else:
            cache_file = cfg.TRAIN.ROIDBS_CACHE_PATH + im_name + \
                pkl_file_ext
          assert os.path.exists(cache_file), \
              'gt roidb data not found at: {}'.format(cache_file)
          with open(cache_file, 'rb') as fid:
            p_db = cPickle.load(fid)
          minibatch_db.append(p_db)
      else:
        minibatch_db = [self._roidb[i] for i in db_inds]
      # put
      blobs = get_minibatch(minibatch_db, self._num_classes)
      self._queue.put(blobs)
