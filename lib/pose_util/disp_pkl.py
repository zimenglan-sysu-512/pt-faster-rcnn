import os
import sys
import cv2
import cPickle
import numpy as np
import scipy.sparse

def dump_pkl(pklobj, out_file):
	with open(out_file, 'wb') as fid:
		cPickle.dump(pklobj, fid, cPickle.HIGHEST_PROTOCOL)

def disp_pkl(in_file):
	assert os.path.exists(in_file), \
			'rpn data not found at: {}'.format(in_file)
	with open(in_file, 'rb') as fid:
		prop = cPickle.load(fid)

	return prop

if __name__ == '__main__':
	''''''	
	in_dir = "/pathTo/"
	file   = "demo.pkl"
	path   = in_dir + file
	prop   = disp_pkl(path)
	prop2  = prop.copy()
	prop["flipped"] = False
	print len(prop['boxes'])
	print prop
	print
	dump_pkl(prop, path)
	prop   = disp_pkl(path)
	print len(prop['boxes'])
	print prop
	print 
	dump_pkl(prop2, path)
	prop   = disp_pkl(path)
	print len(prop['boxes'])
	print prop
	print prop2