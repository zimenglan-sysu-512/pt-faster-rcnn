#!/bin/bash
# Usage:
# 	./pts/torso/VGG16/flic.torso.21/train_torso.sh GPU NET [--set ...]
# Example:
# 	RootDir=~/dongdk/pt-fast_rcnn/
# 	cd $RootDir
# 	./pts/person.torso/VGG16/flic.torso.21/train_torso.sh 0 VGG16 --set RNG_SEED 1701

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}

log_dir="logs/"

exper_dir="person.torso/${NET}/flic.torso.21/"

log_path=$log_dir$exper_dir
mkdir -p $log_path

LOG="${log_path}`date +'%Y-%m-%d_%H-%M-%S'`.log"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

pts="pts/"

imdb="voc_2007_trainval"

cfg="${pts}${exper_dir}train.yml"

# weights="data/imagenet_models/${NET}.v2.caffemodel"
weights="data/faster_rcnn_models/${NET}_faster_rcnn_final.caffemodel"

Others=0

# four stages if Others is set `0`
# 	rpn1 -> fast_rcnn1 -> rpn2 -> fast_rcnn2
# two  stages if Others is set `!=0`
# 	rpn2 -> fast_rcnn2
time ./tools/train_faster_rcnn_alt_opt.py \
  --cfg ${cfg} \
  --imdb ${imdb} \
	--gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights ${weights} \
  --Others $Others \
  ${EXTRA_ARGS}