#block(name=gan_hca, threads=4, memory=16000, subtasks=1, gpus=1, hours=8)
	echo $CUDA_VISIBLE_DEVICES
    #source /home/s7hialtu/anaconda3/etc/profile.d/conda.sh
	#conda activate master-thesis
	echo $1
	python3 $1
