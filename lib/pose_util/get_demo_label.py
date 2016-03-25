#!/usr/bin/env python

import os
import sys
import random

def get_demo_label(in_file, out_file, n_line=30):
	fh1 = open(in_file)
	n_obj = 6
	d_cls = {}
	for line in fh1.readlines():
		line = line.strip()
		info = line.split()
		imgidx, info = info[0], info[1:]
		n_info = len(info)
		assert n_info % n_obj == 0
		for idx in xrange(n_info / n_obj):
			idx2 = idx * n_obj
			cls  = info[idx2 + n_obj - 1]
			cls  = cls.strip() 
			if cls not in d_cls.keys():
				d_cls[cls] = []
			d_cls[cls].append(line)
	fh1.close()
	infos = []
	for cls in d_cls.keys():
		lines = d_cls[cls]
		random.shuffle(lines)
		lines = lines[: min(n_line, len(lines))]
		infos.extend(lines)
	random.shuffle(infos)
	fh2 = open(out_file, "w")
	for info in infos:
		fh2.write(info + "\n")
	fh2.close()

if __name__ == '__main__':
	in_file  = "../labels/train_label.log"
	out_file = "../labels/demo_train.log"
	get_demo_label(in_file, out_file, n_line=40)