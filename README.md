# Master-Thesis

My github repository for my master thesis.

## Setting up Environment

Required packages to run the experiments can be found in ``requirements.txt``. Install the decencies via ``pip`` type:

```bash
pip install -r requirements.txt
```

If you are using ``conda`` run following commands:

```bash
conda env creat -f environment.yml
conda activate master-thesis
```

Additionally, there is a ``docker`` file to setup the environment. First build the docker and rename image with `pytorch_workspace:latest` and run:

```bash
docker run -u $(id -u ${USER}):$(id -g ${USER}) -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes --gpus=all -v "$PWD":/app -v "$ABSOLUTE_PATH_TO_DATASET":/data  -w /app --ipc=host -it pytorch_workspace:latest
```

**Note** To enable jupyter notebook run `jupyter notebook --port=8888 --no-browser --ip=0.0.0.0` inside the docker container.

### Dataset

## Experiments

How to reproduce experiments. TODO
