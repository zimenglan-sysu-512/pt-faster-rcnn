import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil

import add_path
from fast_rcnn.config import cfg, cfg_from_file
from per_tor_util.person_torso_func_v2 import init_net, pose4images_online
from utils.tbox2pbox4pose import _tbox2pbox
import caffe

import cv2
import skimage.io
import time
import pprint
import socket
import os, sys
import argparse
import numpy as np

ROOT_DIRE     = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../..')
UPLOAD_FOLDER = '/home/ddk/download/caffe_demos/'
VIZ_FOLDER    = UPLOAD_FOLDER + 'app_pose_viz/'

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

def create_dire(path):
  if not os.path.isdir(path):
    os.makedirs(path)
  elif not os.path.exists(path):
    os.makedirs(path)
  else:
    pass

# Obtain the flask app object
app = flask.Flask(__name__)

@app.route('/')
def index():
  return flask.render_template('index.html', has_result=False)

@app.route('/image_url', methods=['GET'])
def image_url():
  imageurl = flask.request.args.get('imageurl', '')
  try:
    string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
    image         = caffe.io.load_image(string_buffer)

  except Exception as err:
    # For any exception we encounter in reading the image, we will just not continue.
    logging.info('URL Image open error: %s', err)
    return flask.render_template(
        'index.html', has_result=True, result=(False, 'Cannot open image from URL.')
    )

  logging.info('Image: %s', imageurl)

  filename_ = imageurl.rsplit("/", 1)[1]
  filename  = os.path.join(UPLOAD_FOLDER, filename_)
  skimage.io.imsave(filename, image)

  res = app.clf.run(filename)
  if not res[0]:
    logging.info('URL Image open error (2): %s', res[1])
    return flask.render_template(
        'index.html', has_result=True, result=(False, 'Cannot open image from URL.')
    )
  else:
    _, pt_res, pt_time, out_path, pose_time = res

    viz_im = cv2.imread(out_path)
    result = (True, tuple(pt_res), pt_time, pose_time)

    return flask.render_template('index.html', has_result=True, \
                                result=result, imagesrc=embed_image_html(viz_im))

@app.route('/image_upload', methods=['POST'])
def image_upload():
  try:
    # We will save the file to disk for possible data collection.
    imagefile = flask.request.files['imagefile']
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                werkzeug.secure_filename(imagefile.filename)
    filename  = os.path.join(UPLOAD_FOLDER, filename_)
    imagefile.save(filename)
    logging.info('Saving to %s.', filename)

  except Exception as err:
    logging.info('Uploaded image open error: %s', err)
    return flask.render_template(
        'index.html', has_result=True,
        result=(False, 'Cannot open uploaded image.')
    )

  res = app.clf.run(filename)
  if not res[0]:
    logging.info('Uploaded image open error (2): %s', res[1])
    return flask.render_template(
        'index.html', has_result=True,
        result=(False, 'Cannot open uploaded image.')
    )
  else:
    _, pt_res, pt_time, out_path, pose_time = res

    viz_im = cv2.imread(out_path)
    result = (True, tuple(pt_res), pt_time, pose_time)

    return flask.render_template('index.html', has_result=True, \
                                result=result, imagesrc=embed_image_html(viz_im))

def embed_image_html(image, has_resize=False):
  """Creates an image embedded in HTML base64 format."""
  image2     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image_pil  = Image.fromarray(image2)
  if has_resize:
    image_pil  = image_pil.resize((256, 256))
  string_buf = StringIO.StringIO()
  image_pil.save(string_buf, format='png')
  data       = string_buf.getvalue().encode('base64').replace('\n', '')
  return 'data:image/png;base64,' + data

def allowed_file(filename):
  return (
      '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
  )

def default_args(gpu_id):
  '''args for pose pipeline'''
  args = {}
  args['gpu_id']            = gpu_id
  # person and torso args
  args['pt_choice']         = 0
  args['pt_cls']            = "person"
  args['cfg_file']          = '{}/pts/person.torso/VGG16/per.tor.test/only.21/test.yml'.format(ROOT_DIRE)
  args['cls_filepath']      = '{}/pts/person.torso/pascal_voc_classes_names.filepath'.format(ROOT_DIRE)
  args['pt_def']            = '{}/pts/person.torso/VGG16/per.tor.test/only.21/faster_rcnn_test.pt'.format(ROOT_DIRE)
  args['person_caffemodel'] = '{}/output/person.torso/VGG16/person.only.21/VGG16_faster_rcnn_final.caffemodel'.format(ROOT_DIRE)
  args['torso_caffemodel']  = '{}/output/person.torso/VGG16/torso.only.21/VGG16_faster_rcnn_final.caffemodel'.format(ROOT_DIRE)
  # pose args
  args['pose_def']          = '{}/../pose-caffe/'.format(ROOT_DIRE)
  args['pose_caffemodel']   = '{}/../pose-caffe/'.format(ROOT_DIRE)
  args['max_scale']         = 256
  args['min_sacle']         = 240
  args['pose_dxy']          = 0
  args['torso_ratio']       = 0
  # viz args
  args['draw_text']         = False
  # socket
  args['client_addr']       = "127.0.0.1"
  args['port']              = 8192
  # check
  for key, val in args.iteritems():
    if isinstance(val, basestring) and os.path.isfile(val) and not os.path.exists(val):
      raise Exception("File for {} is missing. Should be at: {}".format(key, val))
  
  return args

class PosePipeline(object):
  ''''''
  def __init__(self, args):
    logging.info('Loading net and associated files...')

    gpu_id = args['gpu_id']
    if gpu_id < 0:
      caffe.set_mode_cpu()
      print "use cpu mode"
    else:
      caffe.set_mode_gpu()
      caffe.set_device(gpu_id)
      cfg.GPU_ID = gpu_id
      print "use gpu mode"
    self.gpu_id = gpu_id
    logging.info('\n\ngpu id: {:s}'.format(str(gpu_id)))
    time.sleep(1)

    cfg_file = args['cfg_file'].strip()
    if os.path.exists(cfg_file) and os.path.isfile(cfg_file):
      cfg_from_file(cfg_file)
    logging.info('\n\ncfg_file: {:s}'.format(cfg_file))

    cls_filepath = args['cls_filepath'].strip()
    if not os.path.exists(cls_filepath) or not os.path.isfile(cls_filepath):
      raise IOError(('{:s} not found.\n').format(cls_filepath))
    with open(cls_filepath) as f:
      classes = [x.strip().lower() for x in f.readlines()]
    self.classes = tuple(classes)
    logging.info('\n\cls_filepath: {:s}'.format(cls_filepath))
    logging.info('\n\nclasses: {:s}'.format(",".join(classes)))

    pt_cls      = args['pt_cls'].strip().split(",")
    self.pt_cls = [cls.strip() for cls in pt_cls]
    logging.info('\n\pt_cls: {:s}'.format(pt_cls))

    self.pt_choice = args['pt_choice']
    logging.info('\n\pt_choice: {:s}'.format(str(self.pt_choice)))

    pt_def = args['pt_def'].strip()
    if not os.path.exists(pt_def) or not os.path.isfile(pt_def):
      raise IOError(('{:s} not found.\n').format(pt_def))
    logging.info('\n\nLoaded pt_def: {:s}'.format(pt_def))

    person_caffemodel      = args['person_caffemodel'].strip()
    if not os.path.exists(person_caffemodel) or not os.path.isfile(person_caffemodel):
      raise IOError(('{:s} not found.\n').format(person_caffemodel))
    self.person_caffemodel = caffe.Net(pt_def, person_caffemodel, caffe.TEST)
    logging.info('\n\nLoaded person network: {:s}'.format(person_caffemodel))

    torso_caffemodel       = args['torso_caffemodel'].strip()
    if not os.path.exists(torso_caffemodel) or not os.path.isfile(torso_caffemodel):
      raise IOError(('{:s} not found.\n').format(torso_caffemodel))
    self.torso_caffemodel  = caffe.Net(pt_def, torso_caffemodel, caffe.TEST)
    logging.info('\n\nLoaded torso network: {:s}'.format(torso_caffemodel))

    init_net(self.person_caffemodel, self.torso_caffemodel)
    logging.info('\n\ninit_net of pt done!')

    # buffer size 
    self.BUF_SIZE = 2048
    # ip and port 
    self.server_addr = ('127.0.0.1', 8192)  

    try : 
      # use TCP and return socket object
      self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error, msg :
      print "Creating Socket Failure. Error Code : " + str(msg[0]) + " Message : " + msg[1]
      sys.exit()
    # connect to server
    self.client.connect(self.server_addr)
    print "\nconnect to", self.server_addr, "\n"

    time.sleep(1)

  def run(self, im_path):
    ''''''
    try:
      image     = cv2.imread(im_path)  
      starttime = time.time()
      pt_res    = pose4images_online(self.person_caffemodel, self.torso_caffemodel, \
                                     image, self.classes, self.pt_cls, choice=self.pt_choice)
      endtime   = time.time()
      pt_time   = '%.3f' % (endtime - starttime)

      # print pt_res
      # send data
      h, w, p_bbox, p_score, t_bbox, t_score = pt_res
      p_bbox2 = _tbox2pbox(t_bbox, w, h, t_ratio=0.78)
      # _. _, p_bbox, _, t_bbox, _ = pt_res
      p_bbox2 = [str(b) for b in p_bbox2]
      t_bbox  = [str(b) for b in t_bbox]
      data    = im_path.strip() + " 0 " + " ".join(p_bbox2).strip() + " " \
                                + " ".join(t_bbox).strip()
      print "data send:", data
      data = self.client.sendall(data)

      # recieve data
      data = self.client.recv(self.BUF_SIZE)
      print "data recieve:", data
      out_path, pose_time = data.split()

      # return True, pt_res, pt_time, im_path, 0 
      return True, pt_res, pt_time, out_path, pose_time 
    except Exception as err:
      logging.info('Person and Torso detection error: %s', err)
      return (False, 'Something went wrong when detect person & torso for the image. Maybe try another one?')

  
def start_tornado(app, port=5000):
  http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
  http_server.listen(port)
  print("Tornado server starting on port {}".format(port))
  tornado.ioloop.IOLoop.instance().start()

def start_from_terminal(app):
  """
  Parse command line options and start the server.
  """
  parser = optparse.OptionParser()
  parser.add_option('-d', '--debug', help="enable debug mode",
      action="store_true", default=False)
  parser.add_option('-p', '--port', help="which port to serve content on",
      type='int', default=5000)
  parser.add_option('-g', '--gpu', help="use gpu mode",
      action='store_true', default=0)

  opts, args    = parser.parse_args()
  pipeline_args = default_args(opts.gpu)

  # Initialize classifier + warm start by forward for allocation
  app.clf = PosePipeline(pipeline_args)

  if opts.debug:
    app.run(debug=True, host='0.0.0.0', port=opts.port)
  else:
    start_tornado(app, opts.port)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  create_dire(UPLOAD_FOLDER)
  create_dire(VIZ_FOLDER)

  start_from_terminal(app)
