#!/usr/bin/env python

import os
import time
import cPickle
import numpy as np
import scipy.sparse
import scipy.io as sio
from fast_rcnn.config import cfg

def is_memmap_numpy_array(array):
  return type(array) == np.core.memmap
 
def does_numpy_path_have_memmap_version(path):
  return not not get_memmap_path_from_numpy_path(path)
 
def get_memmap_path_from_numpy_path(path):
  results = glob.glob('{}*.npmemmap'.format(path))
  if not results:
    return None

  if len(results) != 1:
    raise StandardError("More than one memmap version for file <{}> exists.".format(path))
  return results[0]
 
def create_numpy_memmap(path, dtype_name, shape):
  assert isinstance(dtype_name, basestring)
  assert isinstance(shape, collections.Iterable) or isinstance(shape, int)

  if not is_numpy_memmap_file_path(path):
    path += create_numpy_memmap_path_suffix(dtype_name, shape)

  assert is_numpy_memmap_file_path(path)
  array = np.memmap(path, mode="w+", dtype=dtype_name, shape=shape)
  assert is_memmap_numpy_array(array)
  return array, path
 
def create_numpy_memmap_path_suffix(dtype_name, shape):
  encoded_shape = str(shape)
  if isinstance(shape, collections.Iterable):
    encoded_shape = "-".join(str(i) for i in shape)
  return "_{}.{}.npmemmap".format(dtype_name, encoded_shape)
 
def is_numpy_memmap_file_path(path):
  try:
    parse_numpy_memmap_path(path)
    return True
  except ValueError:
    return False

def parse_numpy_memmap_path(path):
  assert isinstance(path, basestring)
  if not path.endswith(".npmemmap"):
    raise ValueError("Cannot parse a non npmemmap file type, check your path: <{}>".format(path))

  match = re.search(r"([^_]*?).([\d-]*?).npmemmap$", path)
  if match:
    dtype_name = match.group(1)
    packed_shape = match.group(2)
    shape = tuple([int(i) for i in packed_shape.split("-")])
    return dtype_name, shape

  raise ValueError("Could not parse path for numpy memmap information: <{}>".format(path))
 
 
def dump(obj, file_path, use_cpickle=False):
  # Detect if its memmap file, flush / close the file via del before saving
  if is_memmap_numpy_array(obj):
    assert isinstance(obj, np.core.memmap)
    if not is_numpy_memmap_file_path(file_path):
      raise ValueError("Saving a memmap array, but with an invalid filename: <{}>".format(file_path))
    del obj
    return

  if use_cpickle:
    with open(file_path, "wb") as f:
        cPickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
  else:
    # Automatically uses pickle.HIGHEST_PROTOCOL
    joblib.dump(obj, file_path) 
 
def load(file_path, use_cpickle=False, mmap_mode=None):
  if is_numpy_memmap_file_path(file_path):
    dtype_name, shape = parse_numpy_memmap_path(file_path)
    if not mmap_mode:
      mmap_mode='r+' # Default is open writable
    return np.memmap(file_path, mode=mmap_mode, dtype=dtype_name, shape=shape)
  elif use_cpickle:
    # Will automatically know the protocol from the file itself
    with open(file_path, "rb") as f:
        return cPickle.load(f) 
  else:
    return joblib.load(file_path, mmap_mode)
 
 if __name__ == '__main__':
   ''''''
   file_path = ""
   data = load(file_path, mmap_mode='r')
   for idx in xrange(16):
    print 
    print "idx:", idx
    print data[idx]
  print
  print "len of data:", len(data)
  print
  