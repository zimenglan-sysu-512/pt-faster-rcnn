#!/bin/sh

root="/home/ddk/dongdk/pt-fast-rcnn/"
cd $root

log_file=logs/person.torso_VGG16_flic.torso.21_demo.log
rm $log_file

is_video=0

t_cls="person"

im_path="/home/ddk/dongdk/dataset/FLIC/crop.images2/train/"

# write the detected results (bboxes) into file if given
out_file="/home/ddk/dongdk/dataset/FLIC/vision/train.torso/flic_torso_train.txt"	

out_dire="/home/ddk/dongdk/dataset/FLIC/vision/train.torso/train/"
mkdir -p $out_dire

def="${root}/pts/person.torso/VGG16/flic.torso.21/faster_rcnn_test.pt"

cls_filepath="${root}/pts/person.torso/pascal_voc_classes_names.filepath"

cfg_file="${root}/pts/person.torso/VGG16/flic.torso.21/test.yml"

caffemodel="/home/ddk/dongdk/pt-fast-rcnn/output/person.torso/VGG16/flic.torso.21/VGG16_faster_rcnn_final.caffemodel"

# execute
$root/tools/person_torso_demo.py \
		--def $def \
		--t_cls $t_cls \
		--im_path $im_path \
		--is_video $is_video \
		--cfg_file $cfg_file \
		--out_file $out_file \
		--out_dire  $out_dire \
		--caffemodel $caffemodel \
		--cls_filepath $cls_filepath \
		2>&1 | tee -a $log_file