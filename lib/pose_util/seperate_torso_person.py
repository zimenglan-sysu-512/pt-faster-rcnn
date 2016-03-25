#!/usr/bin/env python

import os
import sys
import random

def seperate(in_file, out_dir, n_obj=6, delim=";", prefix="", postfix=".train", file_ext=".log"):
	d_info = {}
	fid = open(in_file)
	for line in fid.readlines():
		line = line.strip()
		info = line.split()
		imgidx = info[0]
		info   = info[1:]
		n_info = len(info)
		assert n_info % n_obj == 0

		d_flag = {}
		for idx in xrange(n_info / n_obj):
			idx2   = idx * n_obj
			s_info = info[idx2: idx2 + n_obj]
			cls    = s_info[-1]
			if cls not in d_info.keys():
				d_info[cls] = ""
			if cls not in d_flag.keys():
				d_flag[cls] = True
			if d_flag[cls]:
				d_flag[cls] = False
				d_info[cls] = d_info[cls] + imgidx
			d_info[cls]   = d_info[cls] + " " + " ".join(s_info)

		for cls in d_info.keys():
			d_info[cls] = d_info[cls] + delim
	fid.close()

	for cls in d_info.keys():
		filepath = out_dir + prefix + cls + postfix + file_ext
		print "cls:", cls
		print "filepath:", filepath
		info = d_info[cls]
		info = info.split(delim)
		fid  = open(filepath, "w")
		random.shuffle(info)
		for line in info:
			line = line.strip()
			if len(line) > 0:
				info = line.split()
				imgidx = info[0]
				info   = info[1:]
				n_info = len(info)
				assert n_info % n_obj == 0
				for idx in xrange(n_info / n_obj):
					idx2       = idx * n_obj
					info[idx2] = str(idx)
				line = imgidx + " " + " ".join(info).strip()
				line = line + "\n"
				fid.write(line)
		fid.close()

if __name__ == '__main__':
	in_file = "../labels/person.torso.2.log"
	out_dir = "../labels/"
	seperate(in_file, out_dir)