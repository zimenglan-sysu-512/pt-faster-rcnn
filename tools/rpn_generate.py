#!/usr/bin/env python

# --------------------------------------------------------
# Fast/er/ R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Generate RPN proposals."""

import _init_paths
import numpy as np
from utils.timer import Timer
from utils import im_util
from utils.im_util import vis_colors
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals, im_proposals
import cPickle
import caffe
import argparse
import pprint
import time, os, cv2, sys

n_vis_colors = len(vis_colors)

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                      default=0, type=int)
  parser.add_argument('--def', dest='prototxt',
                      help='prototxt file defining the network',
                      default=None, type=str)
  parser.add_argument('--caffemodel', dest='caffemodel',
                      help='model to test',
                      default=None, type=str)
  parser.add_argument('--cfg_file', dest='cfg_file',
                      help='optional config file', default=None, type=str)
  parser.add_argument('--wait', dest='wait',
                      help='wait until net file exists',
                      default=True, type=bool)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to test',
                      default='voc_2007_test', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--im_path', dest='im_path',
                      help='input image or file or diretory', 
                      default=None, type=str)
  parser.add_argument('--out_dire', dest='out_dire',
                      help='output diretory', default=None, type=str)
  parser.add_argument('--rpn_type', dest='rpn_type',
                      help='type of rpn to visualize images', 
                      default=0,    type=int)
  parser.add_argument('--top_k', dest='top_k',
                      help='top_k proposals', default=10,   type=int)
  parser.add_argument('--n_rpn_props', dest='n_rpn_props',
                      help='number of rpn proposals', default=2000,   type=int)
  parser.add_argument('--iou', dest='iou',
                      help='iou choice',    default=0,     type=int)
  parser.add_argument('--thresh', dest='thresh',
                      help='iou threshold', default=0.5,   type=float)
  parser.add_argument('--put_text', dest='put_text',
                      help='viz prop',      default=0,     type=int)
  parser.add_argument('--sleep_time', dest='sleep_time',
                      help='time to sleep', default=3,     type=int)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

def _init_net(args):
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.GPU_ID = args.gpu_id

  # RPN test settings
  cfg.TEST.RPN_PRE_NMS_TOP_N  = -1
  cfg.TEST.RPN_POST_NMS_TOP_N = args.n_rpn_props

  print('Using config:')
  pprint.pprint(cfg)

  while not os.path.exists(args.caffemodel) and args.wait:
    print('Waiting for {} to exist...'.format(args.caffemodel))
    time.sleep(10)

  caffe.set_mode_gpu()
  caffe.set_device(args.gpu_id)
  net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
  net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

  return net

def rpn_generate_demo(args):
  ''''''
  net = _init_net(args)

  imdb = get_imdb(args.imdb_name)
  imdb_boxes = imdb_proposals(net, imdb)

  # output_dir = os.path.dirname(args.caffemodel)
  output_dir = get_output_dir(imdb, net)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  rpn_file = os.path.join(output_dir, net.name + '_rpn_proposals.pkl')
  with open(rpn_file, 'wb') as f:
    cPickle.dump(imdb_boxes, f, cPickle.HIGHEST_PROTOCOL)
  print 'Wrote RPN proposals to {}'.format(rpn_file)

def _viz_props(im, im_name, out_dire, boxes, scores, put_text=False):
  ''''''
  assert len(boxes) == len(scores)
  n_boxes = len(boxes)
  for box_i in xrange(n_boxes):
    box   = boxes[box_i]
    box   = [int(b) for b in box]
    score = scores[box_i]
    score = str(score)

    p1    = (box[0], box[1])
    p2    = (box[2], box[3])
    color = vis_colors[box_i % n_vis_colors]
    cv2.rectangle(im, p1, p2, color, 2)
    if put_text:
      p3    = (box[0], box[1] - 5)
      cv2.putText(im, "score: %s" % (score,), p3, \
                  cv2.FONT_HERSHEY_SIMPLEX, .64, color)
  if out_dire is not None:
    out_path = out_dire + im_name
    cv2.imwrite(out_path, im)
  else:
    cv2.imshow(im_name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _nms_props(boxes, scores, top_k, iou, thresh):
  box_c       = 0
  has_in      = []
  area_has_in = []

  inds        = scores.argsort()[::-1]
  n_boxes     = len(inds)
  assert top_k <= n_boxes

  for nms_i in xrange(n_boxes):
    ind = inds[nms_i]
    box = boxes[ind]
    # print "nms_i", nms_i, "ind:", ind

    flag = True
    area = (abs(box[2] - box[0]) + 1) * (abs(box[3] - box[1]) + 1)

    if box_c < top_k:
      if len(has_in) > 0:
        assert len(has_in) == len(area_has_in)
        for j in xrange(len(has_in)):
          ind2  = has_in[j]
          box2  = boxes[ind2]
          area2 = area_has_in[j]

          x1 = np.maximum(box[0], box2[0])
          y1 = np.maximum(box[1], box2[1])
          x2 = np.minimum(box[2], box2[2])
          y2 = np.minimum(box[3], box2[3])
          w  = np.maximum(0., x2 - x1 + 1)
          h  = np.maximum(0., y2 - y1 + 1)
          inter  = w * h

          if iou:
            ovr = inter / (area + 0.)
          else:
            ovr = inter / (area + area2 - inter + 0.)

          # print "area", area, "area2", area2, "inter:", inter, \
          #       "ovr:", ovr
          if ovr > thresh:
            flag = False
            break

    if flag:
      box_c = box_c + 1
      has_in.append(ind)
      area_has_in.append(area)

      if box_c >= top_k:
        break

  topK_boxes  = [boxes[ind]  for ind in has_in]
  topK_scores = [scores[ind] for ind in has_in]
  assert len(topK_boxes) <= top_k
  assert len(topK_boxes) == len(topK_scores)

  return topK_boxes, topK_scores
 
def rpn_generate_ims(args):
  ''''''
  net = _init_net(args)

  if args.im_path is None:
    raise ValueError("im_path can't not be empty")

  im_paths, im_names = im_util.im_paths(args.im_path) 
  
  out_dire = args.out_dire

  top_k    = args.top_k

  thresh   = args.thresh

  iou      = True if args.iou == 0 else False

  put_text = True if args.iou == 0 else False

  print "\n"
  print "top_k:", top_k
  print "thresh:", thresh
  print "iou:", iou
  print "n_rpn_props:", args.n_rpn_props
  print "put_text:",    args.put_text
  print "sleep time:",  args.sleep_time
  print "\n"
  time.sleep(args.sleep_time)

  n_im = len(im_paths)
  for im_i in xrange(n_im):
    timer = Timer()
    timer.tic()

    im_path = im_paths[im_i]
    im_name = im_names[im_i]

    print "im_i:", im_i, "im_path:", im_path
    im = cv2.imread(im_path)

    boxes, scores   = im_proposals(net, im)

    n_boxes = len(boxes)
    boxes   = [list(box) for box in boxes]
    scores  = [score[0] for score in scores]
    scores  = np.array(scores)
    assert len(boxes) == n_boxes
    assert len(boxes) == len(scores)

    boxes2, scores2 = _nms_props(boxes, scores, top_k, iou, thresh)

    _viz_props(im, im_name, out_dire, boxes2, scores2, put_text)

    total_time = timer.toc(average=False)
    print "takes %s to generate props.\n" % (total_time,)

  print "\n\nDone!\n\n"

def run(args):
  if args.rpn_type == 0:
    rpn_generate_demo(args)
    return

  rpn_generate_ims(args)

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  run(args)
