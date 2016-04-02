ROOT_DIR=/home/ddk/dongdk/pt-fast-rcnn

run train
	cd $ROOT_DIR
	./pts/person.torso/VGG16/kinect2.person.21/train.sh 0 VGG16 --set EXP_DIR seed_rng1701 RNG_SEED 1701
	./pts/person.torso/VGG16/kinect2.person.21/train2.sh 0 VGG16 --set EXP_DIR seed_rng1701 RNG_SEED 1701

	./pts/person.torso/VGG16/kinect2.person.21/train.sh 0 VGG16 --set RNG_SEED 1701
	./pts/person.torso/VGG16/kinect2.person.21/train2.sh 0 VGG16 --set RNG_SEED 1701
	
run test
	cd $ROOT_DIR
	cd pts/person.torso/VGG16/kinect2.person.21/
	sh test.sh

poppose
	torso detection