
import os
import cv2
import sys

obj_n  = 6
disp_n = 200
im_ext = ".jpg"
colors = [
		(222,  12,  39), 
		(122, 212, 139), 
		(20,  198,  68), 
		(111,  12, 139), 
		(131, 112, 179), 
		(31,  211,  79), 
		(131, 121, 179), 
		(31,  121, 192), 
		(192,  21,  92), 
		(192,  21, 192), 
		(23,  119, 188), 
		(216, 121,  92), 
		(16,   11,  62), 
		(16,  111, 162), 
		(96,   46,  12), 
]
is_disp   = True
n_color   = len(colors)
cls_color = (23, 119, 188)

def mkdirs(path):
	if not os.path.isdir(path):
		os.makedirs(path)

def im_show(in_dire, out_dire, in_file):
	inst_c = 0
	disp_c = 0
	fh     = open(in_file)

	for line in fh.readlines():
		disp_c += 1
		if disp_c % disp_n == 0:
			print "disp_c:", disp_c

		line = line.strip()
		info = line.split()
		info = [i.strip() for i in info]

		if len(info) <= 0:
			print "error format:", line
			sys.exit(1)
		
		imgidx = info[0].strip()
		info   = info[1:]
		if (len(info) % obj_n) != 0:
			print "error format:", line
			continue
		info_n = len(info)
		n_obj = info_n / obj_n

		im_path = in_dire + imgidx + im_ext
		im      = cv2.imread(im_path)
		h, w, _ = im.shape

		for idx in range(0, n_obj):
			j      = idx * obj_n
			objidx = int(info[j + 0])
			x1     = int(info[j + 1])
			y1     = int(info[j + 2])
			x2     = int(info[j + 3])
			y2     = int(info[j + 4])
			cls    = str(info[j + 5])

			assert idx == objidx, line
			assert x1 >= 1, line
			assert y1 >= 1, line
			assert x2 <= w - 2, line
			assert y2 <= h - 2, line
			assert x1 < x2, line
			assert y1 < y2, line
 			inst_c = inst_c + 1

 			if is_disp:
				p1 = (x1, y1)
				p2 = (x2, y2)
				cv2.rectangle(im, p1, p2, colors[idx % n_color], 2)
				p3 = (x1, y1 - 3)
				cv2.putText(im, cls, p3, cv2.FONT_HERSHEY_SIMPLEX, .6, cls_color)
		if is_disp:
			im_path2 = out_dire + imgidx + im_ext
			cv2.imwrite(im_path2, im)

	# close
	fh.close()

	print "disp_c:", disp_c # 3369
	print "inst_c:", inst_c	# 3975
	print "\n\nDone.\n\n"


if __name__ == '__main__':
	''''''
	in_dire  = "/home/ddk/malong/pt.model/person.torso.dataset.ldp/face/images/"
	out_dire = "/home/ddk/malong/pt.model/person.torso.dataset.ldp/face/vision/V1/"
	in_file  = "/home/ddk/malong/pt.model/person.torso.dataset.ldp/face/labels/face.train.log"

	mkdirs(out_dire)
	im_show(in_dire, out_dire, in_file)