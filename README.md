# Master-Thesis

My github repository for my master thesis. It contains all the experiments except the object detection experiments.

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

Additionally, there is a ``docker`` file to setup the environment.

```bash
docker build -t thesis_workspace .
docker run -u $(id -u ${USER}):$(id -g ${USER}) -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes --gpus=all -v "$PWD":/app -v "$ABSOLUTE_PATH_TO_DATASET":/data  -w /app --ipc=host -it thesis_workspace
```

**Note** To enable jupyter notebook run `jupyter notebook --port=8888 --no-browser --ip=0.0.0.0` inside the docker container.

## Dataset

[Potsdam](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-potsdam/) and [ARTIFIVE](http://rs.ipb.uni-bonn.de/data/artifive-potsdam/) datasets are used. Download the datasets then use ```crop_cars.py``` to crop the cars.

## Experiments

To run the experiments type:

```bash
python -m experiment.<experiment name>
```

See experiments for more detail information.
