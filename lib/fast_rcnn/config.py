# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
  - See tools/{train,test}_net.py for example code that uses cfg_from_file()
  - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

# ###############################################################################################
# 
# Config
# 
# ###############################################################################################
__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# ###############################################################################################
#
# MISC
#
# ###############################################################################################
# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.
# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# For reproducibility
__C.RNG_SEED = 3
# A small number that's used many times
__C.EPS = 1e-14
# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = False
# Default GPU device id
__C.GPU_ID = 0
#######################################################################
__C.ConstChar = "CC"
# 
__C.IS_EXP_DIR = False
__C.IMAGE_EXT = ".jpg"
# For Output Directory
# Place outputs under an experiments directory
__C.EXP_DIR = 'default'
__C.OUTPUT_DIR = "output"
# Classes File Path
__C.CLASSES_FILEPATH = ""
# Prototxt Root Directory
# ROOT_DIR + MODEL_NAME + TASK_NAME + NET + FAST_RCNN_TYPE
__C.TASK_NAME = ""
__C.MODEL_NAME = "models"
__C.FAST_RCNN_TYPE = "faster_rcnn_alt_opt"
# For Debug
__C.INPUT_IMG_DEBUG_NUM = 0
__C.INPUT_DEBUG_STR = "_debug_"
# Default Prototxtes
__C.RPN_TEST = "rpn_test.pt"
__C.MAX_ITERS = "80000,40000,80000,40000"
__C.STAGE1_RPN_SOLVER60K80K = 'stage1_rpn_solver60k80k.pt'
__C.STAGE1_FAST_RCNN_SOLVER30K40K = "stage1_fast_rcnn_solver30k40k.pt"
__C.STAGE2_RPN_SOLVER60K80K = "stage2_rpn_solver60k80k.pt"
__C.STAGE2_FAST_RCNN_SOLVER30K40K = "stage2_fast_rcnn_solver30k40k.pt"
# 
__C.FLIPPED_POSTFIX = "_flipped"
# 
__C.PKL_FILE_EXT = '.pkl'

# ###############################################################################################
#
# Training options
#
# ###############################################################################################
__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
# Default
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# Compute
__C.TRAIN.BBOX_REG_NORMALIZE_MEANS = None
__C.TRAIN.BBOX_REG_NORMALIZE_STDS = None

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = False
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
# 
__C.TRAIN.DATA_BAIS_NUM = 0

# ###################################################
# Human, Face, Torso For Detection
__C.TRAIN.D_INPUT_DIR = ""
__C.TRAIN.D_INPUT_IMG_DIR = ""
__C.TRAIN.D_INPUT_LAB_DIR = ""
__C.TRAIN.D_INPUT_FILE= ""
# For Data Input Directory
__C.TRAIN.DATA = "data"
# For Cache Sub-Directory
__C.TRAIN.CACHE="cache"
# For Display The Shape of Input
__C.TRAIN.P_DISP_IN_SHAPE = 0
# 
__C.TRAIN.IS_RPN_REMOVE_MODEL = True
__C.TRAIN.IS_FAST_RCNN_REMOVE_MODEL = True
# 
__C.TRAIN.RPN_IMS_PER_BATCH = 1
__C.TRAIN.FAST_RCNN_IMS_PER_BATCH = 2
# 
__C.TRAIN.STAGE1_RPN_IS_TRAIN = True
__C.TRAIN.STAGE1_FAST_RCNN_IS_GENERATE_RPN = False
__C.TRAIN.STAGE1_FAST_RCNN_IS_TRAIN = True
__C.TRAIN.STAGE2_RPN_IS_TRAIN = True
__C.TRAIN.STAGE2_FAST_RCNN_IS_GENERATE_RPN = False
__C.TRAIN.STAGE2_FAST_RCNN_IS_TRAIN = True
# 
__C.TRAIN.STAGE1_RPN_MODEL_PATH = ""
__C.TRAIN.STAGE1_RPN_PROPOSAL_PATH = ""
__C.TRAIN.STAGE1_FAST_RCNN_MODEL_PATH = ""
__C.TRAIN.STAGE1_FAST_RCNN_PROPOSAL_PATH = ""
# 
__C.TRAIN.STAGE2_RPN_MODEL_PATH = ""
__C.TRAIN.STAGE2_RPN_PROPOSAL_PATH = ""
__C.TRAIN.STAGE2_FAST_RCNN_MODEL_PATH = ""
__C.TRAIN.STAGE2_FAST_RCNN_PROPOSAL_PATH = ""
# 
# RPOPOSAL_ROIDB_CACHE_DIR + RPOPOSAL_CACHE_SUB_DIR/ROIDB_CACHE_SUB_DIR + RPOPOSAL_ROIDB_STAGE
__C.TRAIN.RPN_CACHE_DIR     = ""
__C.TRAIN.RPN_STAGE1_DIR    = ""
__C.TRAIN.RPN_STAGE2_DIR    = ""
__C.TRAIN.ROIDBS_CACHE_DIR  = ""
__C.TRAIN.ROIDBS_STAGE1_DIR = ""
__C.TRAIN.ROIDBS_STAGE2_DIR = ""
# RPN_CACHE_PATH = RPN_CACHE_DIR + RPN_STAGE_DIR
__C.TRAIN.RPN_CACHE_PATH    = ""
# ROIDBS_CACHE_PATH = ROIDB_CACHE_DIR + ROIDB_STAGE_DIR
__C.TRAIN.ROIDBS_CACHE_PATH = ""

# 
__C.TRAIN.COMP_SIZES_PATH   = ""
__C.TRAIN.COMP_WIDTHS_PATH  = ""
__C.TRAIN.COMP_HEIGHTS_PATH = ""
__C.TRAIN.WIDTHS_PREFIX   = ""
__C.TRAIN.HEIGHTS_PREFIX  = ""
__C.TRAIN.SIZES_PREFIX    = ""
__C.TRAIN.WIDTHS_POSTFIX  = "_widths_of_gt_roidb.pkl"
__C.TRAIN.HEIGHTS_POSTFIX = "_heights_of_gt_roidb.pkl"
__C.TRAIN.SIZES_POSTFIX   = "_sizes_of_gt_roidb.pkl"

# 
__C.TRAIN.PRINT_ITER_NUN = 200


# ###############################################################################################
#
# Testing options
#
# ###############################################################################################
__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
__C.TEST.RPN_POST_NMS_TOP_N_DEFAULT_VAL = 2000  # For training
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# ###################################################
# Human, Face, Torso For Detection
__C.TEST.D_INPUT_DIR = ""
__C.TEST.D_INPUT_IMG_DIR = ""
__C.TEST.D_INPUT_LAB_DIR = ""
__C.TEST.D_INPUT_FILE= ""
# For Data Input Directory
__C.TEST.DATA = "data"
# For Cache Sub-Directory
__C.TEST.CACHE = "cache"
# 
__C.TEST.FASHIONT_HRESHOLD = 0.7
# 
__C.TEST.NMS_THRES  = 0.3
__C.TEST.CONF_THRES = 0.8
__C.TEST.P_NMS_THRES  = 0.3
__C.TEST.P_CONF_THRES = 0.8
__C.TEST.T_NMS_THRES  = 0.3
__C.TEST.T_CONF_THRES = 0.8

# ###############################################################################################
#
# Other Function
#
# ###############################################################################################

def get_output_dir(imdb, net):
  """Return the directory where experimental artifacts are placed.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  path = None
  if __C.IS_EXP_DIR :
    path = osp.abspath(osp.join(__C.ROOT_DIR, __C.OUTPUT_DIR, \
        __C.EXP_DIR))
  else:
    path = osp.abspath(osp.join(__C.ROOT_DIR, __C.OUTPUT_DIR, \
        __C.EXP_DIR, imdb.name))
  print 
  print path
  import time
  time.sleep(3)
  if net is None:
    return path
  else:
    return osp.join(path, net.name)

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.iteritems():
    # a must specify keys that are in b
    if not b.has_key(k):
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                        'for config key: {}').format(type(b[k]),
                                                    type(v), k))
    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print('Error under config key: {}'.format(k))
        raise
    else:
      b[k] = v

def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert d.has_key(subkey)
      d = d[subkey]
    subkey = key_list[-1]
    assert d.has_key(subkey)
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
      type(value), type(d[subkey]))
    d[subkey] = value
