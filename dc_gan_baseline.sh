#block(name=gan_hca, threads=4, memory=16000, subtasks=1, gpus=1, hours=14)
	echo $CUDA_VISIBLE_DEVICES
    source /home/s7hialtu/anaconda3/etc/profile.d/conda.sh
	conda activate master-thesis
	python3 train.py
