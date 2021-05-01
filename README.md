# Master-Thesis

Build the docker and rename image with `pytorch_workspace:latest` and run 

`docker run -u $(id -u ${USER}):$(id -g ${USER}) -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes --gpus=all -v "$PWD":/app -v "$ABSOLUTE_PATH_TO_DATASET":/data  -w /app --ipc=host -it pytorch_workspace:latest`

To enable jupyter notebook run `jupyter notebook --port=8888 --no-browser --ip=0.0.0.0`
