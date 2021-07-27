
#block(name=gan_space_hca, threads=6, memory=60000, subtasks=1, gpus=0, hours=8)
    source /home/s7hialtu/anaconda3/etc/profile.d/conda.sh
	conda activate master-thesis
	python3 gan_space_cube.py /scratch/s7hialtu/dcgan_disc_double_params_padding_reflect_lr_scheduler/lightning_logs/version_0/checkpoints/epoch=899.ckpt /scratch/s7hialtu/gan_space/ --n_samples 20000 --layer conv-1 --n_components 16384
