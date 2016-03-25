# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
# Set
__sets = {}
# Init
__D_INPUT_DIR = ""
__D_INPUT_IMG_DIR = ""
__D_INPUT_LAB_DIR = ""
__D_INPUT_FILE= ""
__data = "data"
__cache = "cache"

import datasets.pascal_voc
import numpy as np

def _selective_search_IJCV_top_k(split, year, top_k, \
    D_INPUT_DIR="", D_INPUT_IMG_DIR="", \
    D_INPUT_LAB_DIR="", D_INPUT_FILE="", \
    data="data", cache="cache"):
  """
  Return an imdb that uses the top k proposals 
  from the selective search IJCV code.
  """
  global __D_INPUT_DIR, __D_INPUT_IMG_DIR
  global __D_INPUT_LAB_DIR, __D_INPUT_FILE
  global __data, __cache
  imdb = datasets.pascal_voc(split, year, \
                  D_INPUT_DIR=__D_INPUT_DIR, \
                  D_INPUT_IMG_DIR=__D_INPUT_IMG_DIR, \
                  D_INPUT_LAB_DIR=__D_INPUT_LAB_DIR, \
                  D_INPUT_FILE=__D_INPUT_FILE, \
                  data=__data, cache=__cache)
  imdb.roidb_handler = imdb.selective_search_IJCV_roidb
  imdb.config['top_k'] = top_k
  return imdb

def _init_sets():
  global __D_INPUT_DIR, __D_INPUT_IMG_DIR
  global __D_INPUT_LAB_DIR, __D_INPUT_FILE
  global __data, __cache

  print "0003"
  print "data:", __data
  print "cache:", __cache
  print 

  for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                datasets.pascal_voc(split, year, \
                    D_INPUT_DIR=__D_INPUT_DIR, \
                    D_INPUT_IMG_DIR=__D_INPUT_IMG_DIR, \
                    D_INPUT_LAB_DIR=__D_INPUT_LAB_DIR, \
                    D_INPUT_FILE=__D_INPUT_FILE, \
                    data=__data, cache=__cache))

  # Set up voc_<year>_<split>_top_<k> using selective search "quality" mode
  # but only returning the first k boxes
  for top_k in np.arange(1000, 11000, 1000):
    for year in ['2007', '2012']:
        for split in ['train', 'val', 'trainval', 'test']:
            name = 'voc_{}_{}_top_{:d}'.format(year, split, top_k)
            __sets[name] = (lambda split=split, year=year, top_k=top_k:
                    _selective_search_IJCV_top_k(split, year, top_k, \
                        D_INPUT_DIR=__D_INPUT_DIR, \
                        D_INPUT_IMG_DIR=__D_INPUT_IMG_DIR, \
                        D_INPUT_LAB_DIR=__D_INPUT_LAB_DIR, \
                        D_INPUT_FILE=__D_INPUT_FILE, \
                        data=__data, cache=__cache))

def get_imdb(name, D_INPUT_DIR="", D_INPUT_IMG_DIR="", \
    D_INPUT_LAB_DIR="", D_INPUT_FILE="", \
    data="data", cache="cache"):
  """Get an imdb (image database) by name."""
  global __D_INPUT_DIR, __D_INPUT_IMG_DIR
  global __D_INPUT_LAB_DIR, __D_INPUT_FILE
  global __data, __cache
  __D_INPUT_DIR = D_INPUT_DIR
  __D_INPUT_IMG_DIR = D_INPUT_IMG_DIR
  __D_INPUT_LAB_DIR = D_INPUT_LAB_DIR
  __D_INPUT_FILE= D_INPUT_FILE
  __data = data
  __cache = cache
  print "0001"
  print "data:", __data
  print "cache:", __cache
  print 

  _init_sets()

  if not __sets.has_key(name):
    raise KeyError('Unknown dataset: {}'.format(name))
  
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  global __D_INPUT_DIR, __D_INPUT_IMG_DIR
  global __D_INPUT_LAB_DIR, __D_INPUT_FILE
  global __data, __cache

  __D_INPUT_DIR = D_INPUT_DIR
  __D_INPUT_IMG_DIR = D_INPUT_IMG_DIR
  __D_INPUT_LAB_DIR = D_INPUT_LAB_DIR
  __D_INPUT_FILE= D_INPUT_FILE
  __data = data
  __cache = cache
  print "0002"
  print "data:", __data
  print "cache:", __cache
  print 

  _init_sets()
  
  print 
  print "list_imdbs keys:"
  print __sets.keys() 
  print
  return __sets.keys()
