#!/bin/sh

root="/home/ddk/malong/crop-fashion-item"
cd $root

log_file=logs/person.torso_VGG16_person.only.21_demo.log
rm $log_file

is_video=0

t_cls="person"

im_path="/home/ddk/malong/dataset/person.torso/demo/mude.images1/"

# out_file="person.bbox.txt"	# write the detected results (bboxes) into file if given

out_dire="/home/ddk/malong/dataset/person.torso/demo/vision/mude.images1/"
mkdir -p $out_dire

def="${root}/pts/person.torso/VGG16/person.only.21/faster_rcnn_test.pt"

cls_filepath="${root}/pts/person.torso/pascal_voc_classes_names.filepath"

cfg_file="${root}/pts/person.torso/VGG16/person.only.21/test.yml"

caffemodel="/home/ddk/malong/pt.model/ldp.faster-rcnn-person-torso-model/person.only.21/VGG16_faster_rcnn_final.caffemodel"

# execute
$root/tools/person_torso_demo.py \
		--def $def \
		--t_cls $t_cls \
		--im_path $im_path \
		--is_video $is_video \
		--cfg_file $cfg_file \
		--out_dire  $out_dire \
		--caffemodel $caffemodel \
		--cls_filepath $cls_filepath \
		2>&1 | tee -a $log_file