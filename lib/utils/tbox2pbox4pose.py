import os
import sys
import cv2
import math
import numpy as np

t_ratio = .76

def _tbox2pbox(t_bbox, w, h):
  # Torso
  x1 = t_bbox[0]
  y1 = t_bbox[1]
  x2 = t_bbox[2]
  y2 = t_bbox[3]
  # Person
  diff_x = x1 - x2
  diff_y = y1 - y2
  t_dit = diff_x * diff_x + diff_y * diff_y
  t_dit = math.sqrt(t_dit)
  t_dit = int(t_dit * t_ratio)
  px1 = x1 - t_dit
  py1 = y1 - t_dit
  px2 = x2 + t_dit
  py2 = y2 + t_dit
  px1 = max(1, px1)
  py1 = max(1, py1)
  px2 = min(w - 2, px2)
  py2 = min(h - 2, py2)
  p_bbox = [px1, py1, px2, py2]
  return p_bbox

def tbox2pbox4pose(in_file, out_file, n_obj=9):
	fh1 = open(in_file)
	fh2 = open(out_file, "w")

	for line in fh1.readlines():
		line = line.strip()
		info = line.split()
		info = [i.strip() for i in info]
		assert len(info) >= 2
		im_path, info = info[0], info[1:]
		im_path       = im_path.strip()
		n_info = len(info)
		assert n_info >= n_obj
		assert n_info % n_obj == 0


		im      = cv2.imread(im_path)
		h, w, _ = im.shape

		res = im_path
		for j in xrange(n_info / n_obj):
			j2 = j * n_obj
			objidx, pbox, tbox = info[j2], info[j2+1: j2+5], info[j2+5: j2+n_obj]
			pbox  = [int(float(c)) for c in pbox]			
			tbox  = [int(float(c)) for c in tbox]			
			pbox2 = _tbox2pbox(tbox, w, h)

			tbox  = [str(c) for c in tbox]
			pbox2 = [str(c) for c in pbox2]
			res   = res + " " + objidx.strip() + " " + \
							" ".join(pbox2).strip() + " " + \
							" ".join(tbox).strip()
		fh2.write(res.strip() + "\n")
	fh1.close()
	fh2.close()

if __name__ == '__main__':
	in_file  = "/home/ddk/download/pose.test.nature.scene/pt_props.txt"
	out_file = "/home/ddk/download/pose.test.nature.scene/pt_props_m.txt"
	tbox2pbox4pose(in_file, out_file)

