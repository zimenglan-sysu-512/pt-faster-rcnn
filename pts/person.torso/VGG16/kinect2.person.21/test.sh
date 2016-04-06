#!/bin/sh

root="/home/ddk/dongdk/pt-fast-rcnn/"
cd $root

log_file=logs/person.torso_VGG16_kinect2.person.21_demo_Kinect2.log
rm $log_file

is_video=0

t_cls="person"

im_path="/home/ddk/dongdk/dataset/Kinect2/labels/up.color2.pbbox.test_pd.logs"

# write the detected results (bboxes) into file if given
out_file="/home/ddk/dongdk/dataset/Kinect2/test_person_results.txt"	

out_dire="/home/ddk/dongdk/dataset/Kinect2/vision/person/"
mkdir -p $out_dire

def="${root}/pts/person.torso/VGG16/kinect2.person.21/faster_rcnn_test.pt"

cls_filepath="${root}/pts/person.torso/pascal_voc_classes_names.filepath"

cfg_file="${root}/pts/person.torso/VGG16/kinect2.person.21/test.yml"

caffemodel="/home/ddk/dongdk/pt-fast-rcnn/output/person.torso/VGG16/kinect2.person.21/VGG16_faster_rcnn_final.caffemodel"

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