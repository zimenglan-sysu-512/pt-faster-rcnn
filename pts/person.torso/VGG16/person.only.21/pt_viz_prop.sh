#!/bin/sh

root="/home/ddk/malong/crop-fashion-item"
cd $root

log_file=logs/person.torso_VGG16_person.only.21_demo_pt.log
rm $log_file


im_path="/home/ddk/cv.life/Nutstore/graduate.project/demo.images/pose0002.jpg"

out_dire="/home/ddk/cv.life/Nutstore/graduate.project/demo.images/props/"
mkdir -p $out_dire

top_k=10

rpn_type=1

thresh=0.63

iou=1

put_text=1

sleep_time=1

n_rpn_props=300

def="${root}/pts/person.torso/VGG16/person.only.21/rpn_test.pt"

cfg_file="${root}/pts/person.torso/VGG16/person.only.21/test.yml"

caffemodel="/home/ddk/malong/pt.model/ldp.faster-rcnn-person-torso-model/person.only.21/VGG16_faster_rcnn_final.caffemodel"

# execute
$root/tools/rpn_generate.py \
		--def $def \
		--iou $iou \
		--top_k $top_k \
		--thresh $thresh \
		--im_path $im_path \
		--rpn_type $rpn_type \
		--cfg_file $cfg_file \
		--put_text $put_text \
		--out_dire  $out_dire \
		--caffemodel $caffemodel \
		--sleep_time $sleep_time \
		--n_rpn_props $n_rpn_props \
		2>&1 | tee -a $log_file