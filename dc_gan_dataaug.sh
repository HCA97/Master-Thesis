#block(name=gan_dataaug_hca, threads=4, memory=16000, subtasks=1, gpus=1, hours=8)
	echo $CUDA_VISIBLE_DEVICES
    source /home/s7hialtu/anaconda3/etc/profile.d/conda.sh
	conda activate master-thesis
	python3 experiment_dataaug.py