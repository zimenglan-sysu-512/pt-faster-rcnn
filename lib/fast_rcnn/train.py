# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import os
import numpy as np
from time import sleep

from caffe.proto import caffe_pb2
import google.protobuf as pb2

class SolverWrapper(object):
  """A simple wrapper around Caffe's solver.
  This wrapper gives us control over he snapshotting process, which we
  use to unnormalize the learned bounding-box regression weights.
  """

  def __init__(self, solver_prototxt, roidb, output_dir, image_index, image_cls,
               pretrained_model=None):
    """Initialize the SolverWrapper."""
    self.image_cls  = image_cls
    self.output_dir = output_dir

    if (cfg.TRAIN.HAS_RPN and cfg.TRAIN.BBOX_REG and cfg.TRAIN.BBOX_NORMALIZE_TARGETS):
      # RPN can only use precomputed normalization because there are no
      # fixed statistics to compute a priori
      assert cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

    if cfg.TRAIN.BBOX_REG:
      print 'computing bounding-box regression targets...'
      print 'in lib/fast_rcnn/train.py -- __init__ func...'
      if roidb is not None:
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(roidb)
      else:
        # bbox reg from cache files
        self.bbox_means = cfg.TRAIN.BBOX_REG_NORMALIZE_MEANS, 
        self.bbox_stds  = cfg.TRAIN.BBOX_REG_NORMALIZE_STDS
        assert (self.bbox_means is not None), 'invalid bbox_means in SolverWrapper'
        assert (self.bbox_stds  is not None), 'invalid bbox_stds in SolverWrapper'

      print 'computing bounding-box regression targets done...'
      print 'in lib/fast_rcnn/train.py -- __init__ func of SolverWrapper class.'
      sleep(3)

    print "instance solver"
    self.solver = caffe.SGDSolver(solver_prototxt)
    if pretrained_model is not None:
      print ('Loading pretrained model weights from {:s}').format(pretrained_model)
      self.solver.net.copy_from(pretrained_model)

    self.solver_param = caffe_pb2.SolverParameter()
    with open(solver_prototxt, 'rt') as f:
      pb2.text_format.Merge(f.read(), self.solver_param)

    print
    print "set image index, image cls and roidb"
    print "in lib/fast_rcnn/train.py ..."
    print 
    self.solver.net.layers[0].set_image_cls(image_cls)
    self.solver.net.layers[0].set_image_index(image_index)
    self.solver.net.layers[0].set_roidb(roidb)
    sleep(3)

  def snapshot(self):
    """Take a snapshot of the network after unnormalizing the learned
    bounding-box regression weights. This enables easy use at test-time.
    """
    net = self.solver.net

    scale_bbox_params = (cfg.TRAIN.BBOX_REG and
                         cfg.TRAIN.BBOX_NORMALIZE_TARGETS and
                         net.params.has_key('bbox_pred'))

    if scale_bbox_params:
      # save original values
      orig_0 = net.params['bbox_pred'][0].data.copy()
      orig_1 = net.params['bbox_pred'][1].data.copy()

      # scale and shift with bbox reg unnormalization; then save snapshot
      net.params['bbox_pred'][0].data[...] = \
              (net.params['bbox_pred'][0].data *
               self.bbox_stds[:, np.newaxis])
      net.params['bbox_pred'][1].data[...] = \
              (net.params['bbox_pred'][1].data *
               self.bbox_stds + self.bbox_means)

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
             if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
    filename = (self.solver_param.snapshot_prefix + infix +
                '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
    filename = os.path.join(self.output_dir, filename)

    net.save(str(filename))
    print 'Wrote snapshot to: {:s}'.format(filename)

    if scale_bbox_params:
      # restore net to original state
      net.params['bbox_pred'][0].data[...] = orig_0
      net.params['bbox_pred'][1].data[...] = orig_1
    return filename

  def train_model(self, max_iters):
    """Network training loop."""
    last_snapshot_iter = -1
    timer = Timer()
    model_paths = []
    while self.solver.iter < max_iters:
      # Make one SGD update
      timer.tic()
      self.solver.step(1)
      timer.toc()
      if self.solver.iter % (10 * self.solver_param.display) == 0:
        print 'speed: {:.3f}s / iter'.format(timer.average_time)

      if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = self.solver.iter
        model_paths.append(self.snapshot())

    if last_snapshot_iter != self.solver.iter:
      print "current iter:", self.solver.iter
      print "last_snapshot_iter:", last_snapshot_iter
      snapshot_path = self.snapshot()
      print "snapshot_path:", snapshot_path
      model_paths.append(snapshot_path)
      sleep(3)
    return model_paths

def get_training_roidb(imdb, roidbs_cache_path=None):
  """Returns a roidb (Region of Interest database) for use in training."""
  print "roidbs_cache_path:", roidbs_cache_path
  if roidbs_cache_path is not None and len(roidbs_cache_path) > 0:
    imdb.config['roidbs_cache_path'] = roidbs_cache_path
    imdb.cache_rpn_roidb()
    return None
  else:
    if cfg.TRAIN.USE_FLIPPED:
      print 'Appending horizontally-flipped training examples...'
      print "In lib/fast_rcnn/train.py -- get_training_roidb func..."
      imdb.append_flipped_images()

      print 'Appending horizontally-flipped training examples done...'
      print 

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir, image_index, image_cls, \
    pretrained_model=None, max_iters=40000):
  """Train a Fast R-CNN network."""
  sw = SolverWrapper(solver_prototxt, roidb, output_dir, image_index, image_cls,
                     pretrained_model=pretrained_model)

  print 'start getting solver...'
  model_paths = sw.train_model(max_iters)
  print 'get solver done'
  sleep(3)
  return model_paths