#!/usr/bin/env python

import os
import time
import cPickle
import numpy as np
import scipy.sparse
import scipy.io as sio


def load_pkl_n(in_file, out_file, out_num):
  print
  print "in_file:", in_file
  print "out_file:", out_file
  print
  assert os.path.exists(in_file), 'pkl data not found at: {}'.format(in_file)

  ## normal loading for pkl file
  # with open(in_file, 'rb') as f:
  #   box_list = cPickle.load(f)

  ## loading by np.memmap for memory reduction
  box_list = np.load(in_file, mmap_mode='r')

  n_box_list = len(box_list)
  out_num = min(out_num, n_box_list)
  box_list2 = box_list[: out_num]
  print "out_num:", out_num
  print "len of box_list2:", len(box_list2)
  print
  with open(out_file, 'wb') as fid:
    cPickle.dump(box_list2, fid, cPickle.HIGHEST_PROTOCOL)
  print 'wrote pkl data to {}'.format(out_file)
  print "done"

if __name__ == "__main__":
  dir = "/home/ubuntu/mydev/output/fashion/VGG16/32-cls-0/"
  in_file = "vgg16_rpn_stage1_iter_200000_proposals.pkl.521326"
  out_file = "vgg16_rpn_stage1_iter_200000_proposals.pkl"
  out_num = 200000
  in_file = dir + in_file
  out_file = dir + out_file
  load_pkl_n(in_file, out_file, out_num)