#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Denke Dong
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
import datasets
import datasets.imdb
from datasets.factory import get_imdb
from fast_rcnn.train  import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from rpn.generate     import imdb_proposals, imdb_proposals2pkls
import pprint
import shutil
import cPickle
import sys, os
import argparse
import numpy as np
import multiprocessing as mp
from time import sleep

def create_dir(path):
  if not os.path.isdir(path):
    os.makedirs(path)  

def check_str_valid(path):
  if path is not None and len(path) > 0:
    return True
  return False

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
  # gpu
  parser.add_argument('--gpu', dest='gpu_id',
                      help='GPU device id to use [0]',
                      default=0, type=int)
  # net name
  parser.add_argument('--net_name', dest='net_name',
                      help='network name (e.g., "ZF")',
                      default=None, type=str)
  # pretrained Model
  parser.add_argument('--weights', dest='pretrained_model',
                      help='initialize with pretrained model weights',
                      default=None, type=str)
  # cfg File
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  # imdb Name
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  # Others
  parser.add_argument('--Others', dest='Others',
                      help='determine how many stages to train, !=0: stage2, 0: stage1&stage2',
                      default=0, type=int)
  # config Keys
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def get_roidb(imdb_name, rpn_file=None, stage_flag=None, roidbs_cache_path=None):
  # Set input info
  data            = cfg.TRAIN.DATA
  cache           = cfg.TRAIN.CACHE
  D_INPUT_DIR     = cfg.TRAIN.D_INPUT_DIR
  D_INPUT_FILE    = cfg.TRAIN.D_INPUT_FILE
  D_INPUT_LAB_DIR = cfg.TRAIN.D_INPUT_LAB_DIR
  D_INPUT_IMG_DIR = cfg.TRAIN.D_INPUT_IMG_DIR
  print 
  print "data:", data
  print "cache:", cache
  if stage_flag:
    print "stage_flag:", stage_flag 
  print

  # get imdb
  imdb = get_imdb(imdb_name, D_INPUT_DIR, D_INPUT_IMG_DIR, \
                  D_INPUT_LAB_DIR, D_INPUT_FILE, data, cache)
  print 'Loaded dataset `{:s}` for training'.format(imdb.name)

  # set roidb handler
  imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
  print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
  
  # rpn File
  if rpn_file is not None:
    imdb.config['rpn_file'] = rpn_file
  
  # get roidb
  roidb = get_training_roidb(imdb, roidbs_cache_path)

  return roidb, imdb

def get_solvers(net_name):
  '''
  Faster R-CNN Alternating Optimization
  '''
  # model name
  model_name     = cfg.MODEL_NAME
  
  # task name
  task_name      = str(cfg.TASK_NAME)
  
  # net name -- one of ["VGG16", "VGG_CNN_M_1024", "ZF", ...]
  fast_rcnn_type = cfg.FAST_RCNN_TYPE
  
  # prototxt
  rpn_test                      = cfg.RPN_TEST
  stage1_rpn_solver60k80k       = cfg.STAGE1_RPN_SOLVER60K80K
  stage1_fast_rcnn_solver30k40k = cfg.STAGE1_FAST_RCNN_SOLVER30K40K
  stage2_rpn_solver60k80k       = cfg.STAGE2_RPN_SOLVER60K80K
  stage2_fast_rcnn_solver30k40k = cfg.STAGE2_FAST_RCNN_SOLVER30K40K
  
  # solver for each training stage
  solvers = None
  if task_name and len(task_name) > 0:
    solvers = [
        [task_name, net_name, fast_rcnn_type, stage1_rpn_solver60k80k],
        [task_name, net_name, fast_rcnn_type, stage1_fast_rcnn_solver30k40k],
        [task_name, net_name, fast_rcnn_type, stage2_rpn_solver60k80k],
        [task_name, net_name, fast_rcnn_type, stage2_fast_rcnn_solver30k40k]]
  else:
    solvers = [
        [net_name, fast_rcnn_type, stage1_rpn_solver60k80k],
        [net_name, fast_rcnn_type, stage1_fast_rcnn_solver30k40k],
        [net_name, fast_rcnn_type, stage2_rpn_solver60k80k],
        [net_name, fast_rcnn_type, stage2_fast_rcnn_solver30k40k]]
  solvers = [os.path.join(cfg.ROOT_DIR, model_name, *s) for s in solvers]
  
  # iterations for each training stage
  max_iters = cfg.MAX_ITERS
  max_iters = max_iters.strip()
  if max_iters is not None and len(max_iters) > 1:
    max_iters = max_iters.split(",")
    max_iters = [int(mi.strip()) for mi in max_iters]
  else: # Default
    max_iters = [80000, 40000, 80000, 40000]
  print "max_iters:", max_iters
  if len(max_iters) != len(solvers):
    raise IOError("max_iters has been set incorrectedly...")

  # test prototxt for the RPN
  rpn_test_prototxt = None
  if task_name and len(task_name) > 0:
    rpn_test_prototxt = os.path.join(
        cfg.ROOT_DIR, model_name, task_name, net_name, fast_rcnn_type, rpn_test)
  else:
    rpn_test_prototxt = os.path.join(
        cfg.ROOT_DIR, model_name, net_name, fast_rcnn_type, rpn_test)

  # info
  for solver in solvers:
    print "solver:", solver
  print "rpn_test_prototxt:", rpn_test_prototxt
  sleep(3) 
  
  return solvers, max_iters, rpn_test_prototxt

# #########################################################################
#     Training Progress
# #########################################################################

def _init_caffe(cfg):
  """Initialize pycaffe in a training process.
  """
  import caffe
  
  # fix the random seeds (numpy and caffe) for reproducibility
  np.random.seed(cfg.RNG_SEED)
  caffe.set_random_seed(cfg.RNG_SEED)

  # set up caffe
  caffe.set_mode_gpu()
  caffe.set_device(cfg.GPU_ID)
  print "set gpu mode done, where gpu id:", cfg.GPU_ID
  sleep(3) 

def train_rpn(queue=None, imdb_name=None, init_model=None, \
    solver=None, max_iters=None, cfg=None):
  """
  Train a Region Proposal Network in a separate training process.
  """
  # Not using any proposals, just ground-truth boxes
  cfg.TRAIN.HAS_RPN = True
  # Applies only to Fast R-CNN bbox regression
  cfg.TRAIN.BBOX_REG = False  
  # Roidb handler
  cfg.TRAIN.PROPOSAL_METHOD = 'gt'
  # Batch size
  cfg.TRAIN.IMS_PER_BATCH = cfg.TRAIN.RPN_IMS_PER_BATCH
  
  print 'Init model: {}'.format(init_model)
  print('Using config:')
  pprint.pprint(cfg)

  import caffe
  _init_caffe(cfg)

  # Roidb & Imdb
  stage_flag  = "RPN TRAIN"
  roidb, imdb = get_roidb(imdb_name, stage_flag=stage_flag)
  print 'roidb len: {}'.format(len(roidb))

  # Output directory
  output_dir = get_output_dir(imdb, None)
  print 'Output will be saved to `{:s}`'.format(output_dir)

  # Net & Solver
  # here set image_cls param to be None
  # since rpn training need all images of input dataset
  # but it remains the data bias
  model_paths = train_net(solver, roidb, output_dir,
                          imdb.image_index, None,
                          pretrained_model=init_model,
                          max_iters=max_iters)

  # Cleanup all but the final model
  if cfg.TRAIN.IS_RPN_REMOVE_MODEL:
    for i in model_paths[:-1]:
      if os.path.exists(i) and os.path.isfile(i):
        os.remove(i)
  rpn_model_path = model_paths[-1]

  # Send final model path through the multiprocessing queue
  queue.put({'model_path': rpn_model_path})

def rpn_generate(queue=None, imdb_name=None, rpn_model_path=None, cfg=None, \
    rpn_test_prototxt=None, rpn_cache_path=None):
  """
  Use a trained RPN to generate proposals.
  """
  print 'RPN model: {}'.format(rpn_model_path)
  print 
  # no pre NMS filtering
  cfg.TEST.RPN_PRE_NMS_TOP_N = -1     
  # limit top boxes after NMS
  cfg.TEST.RPN_POST_NMS_TOP_N = cfg.TEST.RPN_POST_NMS_TOP_N_DEFAULT_VAL  
  print('Using config:')
  pprint.pprint(cfg)

  import caffe
  _init_caffe(cfg)

  # NOTE: the matlab implementation computes proposals on flipped images, too.
  # We compute them on the image once and then flip the already computed
  # proposals. This might cause a minor loss in mAP (less proposal jittering).
  data            = cfg.TRAIN.DATA
  cache           = cfg.TRAIN.CACHE
  D_INPUT_DIR     = cfg.TRAIN.D_INPUT_DIR
  D_INPUT_FILE    = cfg.TRAIN.D_INPUT_FILE
  D_INPUT_LAB_DIR = cfg.TRAIN.D_INPUT_LAB_DIR
  D_INPUT_IMG_DIR = cfg.TRAIN.D_INPUT_IMG_DIR
  stage_flag      = "GENERATE RPN PROPOSALS"
  print 
  print "data:", data
  print "cache:", cache
  print "stage_flag:", stage_flag
  print 
  sleep(3)
  
  # Instance
  imdb = get_imdb(imdb_name, D_INPUT_DIR, D_INPUT_IMG_DIR, \
                  D_INPUT_LAB_DIR, D_INPUT_FILE, data, cache)
  print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

  # Output directory
  output_dir = get_output_dir(imdb, None)
  print 'Output will be saved to `{:s}`'.format(output_dir)
  
  print "\n\n"
  # Network -> Load RPN & Configurations
  print "rpn generate -> load trained model from:"
  print "  pt:", rpn_test_prototxt
  print "  model:", rpn_model_path
  print "\n\n"
  sleep(3)
  rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)
  sleep(3)

  # Get proposals from rpn network and write them into files
  if rpn_cache_path is not None and len(rpn_cache_path) > 0:
    print "use cache strategy for generating props -- per image per props file"
    print "start writing props into a file for each image"
    imdb_proposals2pkls(rpn_net, imdb, rpn_cache_path)
    print "finish writing props into a file for each image"
    print "and set rpn proposals path into queue"
    rpn_proposals_path = rpn_cache_path
    queue.put({'proposal_path': rpn_proposals_path})
  else:
    print "use whole strategy for generating props -- all images one props file"
    print "generate proposals on the imdb"
    rpn_proposals = imdb_proposals(rpn_net, imdb)
    print "write proposals to disk and send the proposal file path through the"
    print "multiprocessing queue"
    rpn_net_name  = os.path.splitext(os.path.basename(rpn_model_path))[0]
    rpn_proposals_path = os.path.join(
        output_dir, rpn_net_name + '_proposals.pkl')
    # write into file 
    with open(rpn_proposals_path, 'wb') as f:
      cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)
    # put into queue
    queue.put({'proposal_path': rpn_proposals_path})
  sleep(3)

def train_fast_rcnn(queue=None, imdb_name=None, init_model=None, solver=None, \
    max_iters=None, cfg=None, rpn_file=None, roidbs_cache_path=None):
  """
  Train a Fast R-CNN using proposals generated by an RPN.
  """
  # not generating prosals on-the-fly
  cfg.TRAIN.HAS_RPN = False           
  # use pre-computed RPN proposals instead
  if roidbs_cache_path is not None and len(roidbs_cache_path) > 0:
    cfg.TRAIN.PROPOSAL_METHOD = 'cache_rpn'
  else:
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'
  cfg.TRAIN.IMS_PER_BATCH = cfg.TRAIN.FAST_RCNN_IMS_PER_BATCH

  print 'Init model: {}'.format(init_model)
  print 'RPN proposals: {}'.format(rpn_file)
  print 'roidbs_cache_path: {}'.format(roidbs_cache_path)
  print 'Using config:'
  print 
  pprint.pprint(cfg)

  import caffe
  _init_caffe(cfg)

  # get roidb 
  stage_flag  = "FAST_RCNN TRAIN"
  roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file, stage_flag=stage_flag, \
                          roidbs_cache_path=roidbs_cache_path)
  # get output
  output_dir  = get_output_dir(imdb, None)
  print 'Output will be saved to `{:s}`'.format(output_dir)
  # Train Fast R-CNN
  model_paths = train_net(solver, roidb, output_dir,
                          imdb.image_index, imdb.image_cls,
                          pretrained_model=init_model, max_iters=max_iters)
  # Cleanup all but the final model
  if cfg.TRAIN.IS_FAST_RCNN_REMOVE_MODEL:
    for i in model_paths[:-1]:
      os.remove(i)
  fast_rcnn_model_path = model_paths[-1]
  # Send Fast R-CNN model path over the multiprocessing queue
  queue.put({'model_path': fast_rcnn_model_path})

# #########################################################################
#     Stage 0 init
# #########################################################################

def stage0_init_vars(args):
  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)
  cfg.GPU_ID = args.gpu_id

  # queue for communicated results between processes
  mp_queue = mp.Queue()

  # solves, iters, etc. for each training stage
  solvers, max_iters, rpn_test_prototxt = get_solvers(args.net_name)

  return mp_queue, solvers, max_iters, rpn_test_prototxt

# #########################################################################
#     Stage 1 rpn & fast_rcnn
# #########################################################################

def stage1_train_rpn(mp_queue, args, solver, max_iter):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 1 RPN, init from ImageNet model'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
  mp_kwargs = dict(
          queue=mp_queue,
          imdb_name=args.imdb_name,
          init_model=args.pretrained_model,
          solver=solver,
          max_iters=max_iter,
          cfg=cfg)

  p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
  p.start()
  rpn_stage1_out = mp_queue.get()
  p.join()
  
  return mp_queue, rpn_stage1_out

def stage1_rpn_generate_props(mp_queue, args, rpn_test_prototxt, rpn_stage1_out):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 1 RPN, generate proposals'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 
  rpn_cache_path = None
  cfg.TRAIN.RPN_CACHE_PATH = cfg.TRAIN.RPN_CACHE_DIR + \
      cfg.TRAIN.RPN_STAGE1_DIR
  if check_str_valid(cfg.TRAIN.RPN_CACHE_PATH):
    rpn_cache_path = cfg.TRAIN.RPN_CACHE_PATH
    create_dir(rpn_cache_path)
  print "rpn_cache_path:", rpn_cache_path
  print "rpn model path:", rpn_stage1_out['model_path']

  mp_kwargs = dict(
          queue=mp_queue,
          imdb_name=args.imdb_name,
          rpn_model_path=str(rpn_stage1_out['model_path']),
          cfg=cfg,
          rpn_test_prototxt=rpn_test_prototxt,
          rpn_cache_path=rpn_cache_path)

  p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
  p.start()
  rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
  p.join()

  return mp_queue, rpn_stage1_out

def stage1_train_fast_rcnn(mp_queue, args, solver, max_iter, rpn_stage1_out):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'

  roidbs_cache_path = None
  cfg.TRAIN.ROIDBS_CACHE_PATH = cfg.TRAIN.ROIDBS_CACHE_DIR + \
      cfg.TRAIN.ROIDBS_STAGE1_DIR
  if check_str_valid(cfg.TRAIN.ROIDBS_CACHE_PATH):
    roidbs_cache_path = cfg.TRAIN.ROIDBS_CACHE_PATH
    create_dir(roidbs_cache_path)
  print "roidbs_cache_path:", roidbs_cache_path
  print "proposals path:", rpn_stage1_out['proposal_path']

  mp_kwargs = dict(
          queue=mp_queue,
          imdb_name=args.imdb_name,
          init_model=args.pretrained_model,
          solver=solver,
          max_iters=max_iter,
          cfg=cfg,
          rpn_file=rpn_stage1_out['proposal_path'],
          roidbs_cache_path=roidbs_cache_path)
  
  p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
  p.start()
  fast_rcnn_stage1_out = mp_queue.get()
  p.join()
  
  return mp_queue, fast_rcnn_stage1_out

# #########################################################################
#     Stage 2 rpn & fast_rcnn
# #########################################################################

def stage2_train_rpn(mp_queue, args, solver, max_iter, fast_rcnn_stage1_out):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 2 RPN, init from stage 1 Fast R-CNN model'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'

  mp_kwargs = dict(
          queue=mp_queue,
          imdb_name=args.imdb_name,
          init_model=str(fast_rcnn_stage1_out['model_path']),
          solver=solver,
          max_iters=max_iter,
          cfg=cfg)
  
  p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
  p.start()
  rpn_stage2_out = mp_queue.get()
  p.join()
  
  return mp_queue, rpn_stage2_out

def stage2_rpn_generate_props(mp_queue, args, rpn_test_prototxt, rpn_stage2_out):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 2 RPN, generate proposals'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 
  rpn_cache_path = None
  cfg.TRAIN.RPN_CACHE_PATH = cfg.TRAIN.RPN_CACHE_DIR + \
      cfg.TRAIN.RPN_STAGE2_DIR
  if check_str_valid(cfg.TRAIN.RPN_CACHE_PATH):
    rpn_cache_path = cfg.TRAIN.RPN_CACHE_PATH
    create_dir(rpn_cache_path)
  print "rpn_cache_path:", rpn_cache_path
  print "rpn model path:", rpn_stage2_out['model_path']

  mp_kwargs = dict(
          queue=mp_queue,
          imdb_name=args.imdb_name,
          rpn_model_path=str(rpn_stage2_out['model_path']),
          cfg=cfg,
          rpn_test_prototxt=rpn_test_prototxt,
          rpn_cache_path=rpn_cache_path)

  p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
  p.start()
  rpn_stage2_out['proposal_path'] = mp_queue.get()['proposal_path']
  p.join() 
  
  return mp_queue, rpn_stage2_out

def stage2_train_fast_rcnn(mp_queue, args, solver, max_iter, rpn_stage2_out):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'

  roidbs_cache_path = None
  cfg.TRAIN.ROIDBS_CACHE_PATH = cfg.TRAIN.ROIDBS_CACHE_DIR + \
      cfg.TRAIN.ROIDBS_STAGE2_DIR
  if check_str_valid(cfg.TRAIN.ROIDBS_CACHE_PATH):
    roidbs_cache_path = cfg.TRAIN.ROIDBS_CACHE_PATH
    create_dir(roidbs_cache_path)
  print "roidbs_cache_path:", roidbs_cache_path
  print "proposals path:", rpn_stage2_out['proposal_path']

  mp_kwargs = dict(
          queue=mp_queue,
          imdb_name=args.imdb_name,
          init_model=str(rpn_stage2_out['model_path']),
          solver=solver,
          max_iters=max_iter,
          cfg=cfg,
          rpn_file=rpn_stage2_out['proposal_path'],
          roidbs_cache_path=roidbs_cache_path)
  
  p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
  p.start()
  fast_rcnn_stage2_out = mp_queue.get()
  p.join()
  
  return mp_queue, fast_rcnn_stage2_out

# #########################################################################
#     Stage 1 & 2 & 3 in pipeline
# #########################################################################

def stage1_rpn_fast_rcnn_train(args, mp_queue, solvers, max_iters, \
    rpn_test_prototxt):
  ''''''
  rpn_stage1_out = None
  fast_rcnn_stage1_out = None
  
  # RPN
  if cfg.TRAIN.STAGE1_RPN_IS_TRAIN:
    print "print set widths_heights_sizes_prefix"
    widths_heights_sizes_prefix = "rpn_stage1_"
    datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)
    
    print "start stage 1 of rpn training progress"
    mp_queue, rpn_stage1_out = stage1_train_rpn(mp_queue, args, \
        solvers[0], max_iters[0])
    print "finish stage 1 of rpn training progress"
    print "stage1 -- rpn model_path:", rpn_stage1_out['model_path']
    sleep(3)

    print 
    print "print set widths_heights_sizes_prefix"
    widths_heights_sizes_prefix = "rpn_generate_stage1_"
    datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

    print "start stage 1 of generating proposals using trained rpn model"
    mp_queue, rpn_stage1_out = stage1_rpn_generate_props(mp_queue, args, \
        rpn_test_prototxt, rpn_stage1_out)
    print "finish stage 1 of generating proposals using trained rpn model"
    print "stage1 -- generated proposal_path:", rpn_stage1_out['proposal_path']
    sleep(3)
  else:
    rpn_stage1_out = {}
    # model path
    if cfg.TRAIN.STAGE1_RPN_MODEL_PATH is not None and \
        len(cfg.TRAIN.STAGE1_RPN_MODEL_PATH) > 0:
      rpn_stage1_out['model_path']    = cfg.TRAIN.STAGE1_RPN_MODEL_PATH
    else:
      raise KeyError('stage1 - rpn model path does not exist')

    rpn_cache_path = None
    cfg.TRAIN.RPN_CACHE_PATH = cfg.TRAIN.RPN_CACHE_DIR + \
        cfg.TRAIN.RPN_STAGE1_DIR
    if check_str_valid(cfg.TRAIN.RPN_CACHE_PATH):
      rpn_cache_path = cfg.TRAIN.RPN_CACHE_PATH
    # proposal path
    if cfg.TRAIN.STAGE1_RPN_PROPOSAL_PATH is not None and \
        len(cfg.TRAIN.STAGE1_RPN_PROPOSAL_PATH) > 0:
      rpn_stage1_out['proposal_path'] = cfg.TRAIN.STAGE1_RPN_PROPOSAL_PATH
    elif rpn_cache_path is not None and len(rpn_cache_path) > 0:
      rpn_stage1_out['proposal_path'] = rpn_cache_path
    else:
      raise KeyError('stage1 - rpn proposal path does not exist')
  print
  print rpn_stage1_out
  print
  
  # Fast RCNN
  if cfg.TRAIN.STAGE1_FAST_RCNN_IS_TRAIN:
    # need proposals generating?
    if cfg.TRAIN.STAGE1_FAST_RCNN_IS_GENERATE_RPN:
      print 
      print "print set widths_heights_sizes_prefix"
      widths_heights_sizes_prefix = "rpn_generate_stage1_"
      datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

      print "start stage 1 of generating proposals using trained rpn model"
      mp_queue, rpn_stage1_out = stage1_rpn_generate_props(mp_queue, args, \
        rpn_test_prototxt, rpn_stage1_out)
      print "finish stage 1 of generating proposals using trained rpn model"
      print "stage1 -- generated proposal_path:", rpn_stage1_out['proposal_path']
      sleep(3)

    print 
    print "print set widths_heights_sizes_prefix"
    widths_heights_sizes_prefix = "fast_rcnn_stage1_"
    datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

    print "start stage 1 of fast_rcnn training progress"
    mp_queue, fast_rcnn_stage1_out = stage1_train_fast_rcnn(mp_queue, args, \
        solvers[1], max_iters[1], rpn_stage1_out)
    print "finish stage 1 of fast_rcnn training progress"
    print "stage1 -- fast_rcnn model_path:", fast_rcnn_stage1_out['model_path']
    sleep(3)
  else:
    fast_rcnn_stage1_out = {}
    fast_rcnn_stage1_out['model_path'] = cfg.TRAIN.STAGE1_FAST_RCNN_MODEL_PATH

  return rpn_stage1_out, fast_rcnn_stage1_out

def stage2_rpn_fast_rcnn_train(args, mp_queue, solvers, max_iters, \
    rpn_test_prototxt, fast_rcnn_stage1_out):
  rpn_stage2_out = None
  fast_rcnn_stage2_out = None
  
  # RPN
  if cfg.TRAIN.STAGE2_RPN_IS_TRAIN:
    print 
    print "print set widths_heights_sizes_prefix"
    widths_heights_sizes_prefix = "rpn_stage2_"
    datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

    print "start stage 2 of rpn training progress"
    mp_queue, rpn_stage2_out = stage2_train_rpn(mp_queue, args, \
       solvers[2], max_iters[2], fast_rcnn_stage1_out)
    print "finish stage 2 of rpn training progress"
    print "stage2 -- rpn model_path:", rpn_stage2_out['model_path']
    sleep(3)

    print 
    print "print set widths_heights_sizes_prefix"
    widths_heights_sizes_prefix = "rpn_generate_stage2_"
    datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

    print "start stage 2 of generating proposals using trained rpn model"
    mp_queue, rpn_stage2_out = stage2_rpn_generate_props(mp_queue, args, \
       rpn_test_prototxt, rpn_stage2_out)
    print "finish stage 2 of generating proposals using trained rpn model"
    print "stage2 -- generated proposal_path:", rpn_stage2_out['proposal_path']
    sleep(3)
  else:
    rpn_stage2_out = {}
    # model path
    if cfg.TRAIN.STAGE2_RPN_MODEL_PATH is not None and \
        len(cfg.TRAIN.STAGE2_RPN_MODEL_PATH) > 0:
      rpn_stage2_out['model_path']    = cfg.TRAIN.STAGE2_RPN_MODEL_PATH
    else:
      raise KeyError('stage2 - rpn model path does not exist')

    rpn_cache_path = None
    cfg.TRAIN.RPN_CACHE_PATH = cfg.TRAIN.RPN_CACHE_DIR + \
        cfg.TRAIN.RPN_STAGE2_DIR
    if check_str_valid(cfg.TRAIN.RPN_CACHE_PATH):
      rpn_cache_path = cfg.TRAIN.RPN_CACHE_PATH
    # proposal path
    if cfg.TRAIN.STAGE2_RPN_PROPOSAL_PATH is not None and \
        len(cfg.TRAIN.STAGE2_RPN_PROPOSAL_PATH) > 0:
      rpn_stage2_out['proposal_path'] = cfg.TRAIN.STAGE2_RPN_PROPOSAL_PATH
    elif rpn_cache_path is not None and len(rpn_cache_path) > 0:
      rpn_stage2_out['proposal_path'] = rpn_cache_path
    else:
      raise KeyError('stage2 - rpn proposal path does not exist')

  # Fast RCNN
  if cfg.TRAIN.STAGE2_FAST_RCNN_IS_TRAIN:
    # need proposals generating?
    if cfg.TRAIN.STAGE2_FAST_RCNN_IS_GENERATE_RPN:
      print 
      print "print set widths_heights_sizes_prefix"
      widths_heights_sizes_prefix = "rpn_generate_stage2_"
      datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

      print "start stage 2 of generating proposals using trained rpn model"
      mp_queue, rpn_stage2_out = stage2_rpn_generate_props(mp_queue, args, \
        rpn_test_prototxt, rpn_stage2_out)
      print "finish stage 2 of generating proposals using trained rpn model"
      print "stage2 -- generated proposal_path:", rpn_stage2_out['proposal_path']
      sleep(3)

    print 
    widths_heights_sizes_prefix = "fast_rcnn_stage2_"
    datasets.imdb.widths_heights_sizes_prefix(widths_heights_sizes_prefix)

    print "start stage 2 of fast_rcnn training progress"
    mp_queue, fast_rcnn_stage2_out = stage2_train_fast_rcnn(mp_queue, args, \
       solvers[3], max_iters[3], rpn_stage2_out)
    print "finish stage 2 of fast_rcnn training progress"
    print "stage2 -- fast_rcnn model_path:", fast_rcnn_stage2_out['model_path']
    sleep(3)
  else:
    fast_rcnn_stage2_out               = {}
    fast_rcnn_stage2_out['model_path'] = cfg.STAGE2_FAST_RCNN_MODEL_PATH

  return rpn_stage2_out, fast_rcnn_stage2_out

def stage3_save_final_model(args, fast_rcnn_stage2_out):
  print 
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  print 'Stage 3 Create final model (just a copy of the last stage)'
  print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
  final_path = os.path.join(
          os.path.dirname(fast_rcnn_stage2_out['model_path']),
          args.net_name + '_faster_rcnn_final.caffemodel')
  print 'cp {} -> {}'.format(
          fast_rcnn_stage2_out['model_path'], final_path)
  shutil.copy(fast_rcnn_stage2_out['model_path'], final_path)
  print 'Final model: {}'.format(final_path)

# #########################################################################
#     Training Progress
# #########################################################################

# Stage 1 -- train rpn | rnp generating props | train fast_rcnn
# Stage 2 -- train rpn | rnp generating props | train fast_rcnn
def Origin_Train(args):
  ''''''
  # Stage 0
  mp_queue, solvers, max_iters, rpn_test_prototxt = stage0_init_vars(args)

  # Stage 1
  rpn_stage1_out, fast_rcnn_stage1_out = stage1_rpn_fast_rcnn_train(args, \
      mp_queue, solvers, max_iters, rpn_test_prototxt)

  # Stage 2
  _, fast_rcnn_stage2_out = stage2_rpn_fast_rcnn_train(args, mp_queue, \
      solvers, max_iters, rpn_test_prototxt, fast_rcnn_stage1_out)

  # Stage 3 
  print "start stage 3 of save the trained models of rpn and fast_rcnn"
  stage3_save_final_model(args, fast_rcnn_stage2_out)
  print "finish stage 3 of save the trained models of rpn and fast_rcnn"

# Stage 2 -- train rpn | rnp generating props | train fast_rcnn
def Others_Task_Train(args):
  ''''''
  # Stage 0
  fast_rcnn_stage1_out = {}
  fast_rcnn_stage1_out["model_path"] = args.pretrained_model
  mp_queue, solvers, max_iters, rpn_test_prototxt = stage0_init_vars(args)

  # Stage 2
  _, fast_rcnn_stage2_out = stage2_rpn_fast_rcnn_train(args, mp_queue, \
        solvers, max_iters, rpn_test_prototxt, fast_rcnn_stage1_out)

  # Stage 3
  print "start stage 3 of save the trained models of rpn and fast_rcnn"
  stage3_save_final_model(args, fast_rcnn_stage2_out)
  print "finish stage 3 of save the trained models of rpn and fast_rcnn"

# #########################################################################
#     Run
#     Pycaffe doesn't reliably free GPU memory when instantiated nets are
#     discarded (e.g. "del net" in Python code). To work around this issue, 
#     each training stage is executed in a separate process using
#     multiprocessing.Process.
# #########################################################################

def run():
  '''
  Train Faster RCNN For Different Vision Tasks.
  '''
  # Get Args
  args = parse_args()

  if args.Others != 0:
    print 
    print "###############################"
    print "Others_Task_Train"
    print 
    sleep(3)
    Others_Task_Train(args)
  else:
    print 
    print "###############################"
    print "Origin_Train"
    print 
    sleep(3)
    Origin_Train(args)

  print
  print "Training Done."
  print

if __name__ == '__main__':
  run()