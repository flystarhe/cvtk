# Semantic segmentation reference training scripts
Tested on `pytorch:1.8.1`.

## docker
* python: 3.8
* pytorch: 1.8.1

```sh
docker pull flystarhe/torch:1.8.1-cuda10.2-dev
docker tag flystarhe/torch:1.8.1-cuda10.2-dev torch:1.8.1-cuda10.2-dev

docker save -o torch1.8.1-cuda10.2-dev.tar torch:1.8.1-cuda10.2-dev
docker load -i torch1.8.1-cuda10.2-dev.tar

n=torch18_cu10
t=torch:1.8.1-cuda10.2-dev
docker run --gpus all -d -p 7000:9000 --ipc=host --name ${n} -v "$(pwd)"/${n}:/workspace ${t} ssh
```

## training scripts
Assume `git clone https://github.com/flystarhe/cvtk.git /workspace/cvtk`. you must modify the following flags:

* `--nproc_per_node=<number_of_gpus_available>`
* `fcn_resnet50,deeplabv3_resnet50,deeplabv3_mobilenet_v3_large,lraspp_mobilenet_v3_large`

```jupyter
import os
import time

CVTK_HOME = "/workspace/cvtk"
!cd {CVTK_HOME} && git log -1 --oneline
#os.environ["MKL_THREADING_LAYER"] = "GNU"

EPOCHS = 90
EXPERIMENT_NAME = time.strftime("T%m%d")
DATA_PATH = "/workspace/notebooks/coco_dataset"
ARGS = " ".join([
    "-m torch.distributed.launch --nproc_per_node=2 --use_env",
    f"{CVTK_HOME}/references/segmentation/train.py",
    f"--data-path {DATA_PATH}",
    "--train coco.json",
    "--val coco.json",
    "--single-cls",
    "--max-size 512",
    "--crop-size 480",
    "--model fcn_resnet50",
    "--aux-loss",
    "--loss-fn topk",
    f"--epochs {EPOCHS}",
    "-b 8",
    "-j 16",
    "--lr 0.01",
    "--print-freq 10",
    f"--output-dir /workspace/results/{EXPERIMENT_NAME}",
])
!PYTHONPATH={CVTK_HOME} python {ARGS}
```

## test scripts
```
#EPOCHS = 90
#EXPERIMENT_NAME = "output_dir_from_training"
#DATA_PATH = "/workspace/notebooks/coco_dataset"
ARGS = " ".join([
    "-m torch.distributed.launch --nproc_per_node=2 --use_env",
    f"{CVTK_HOME}/references/segmentation/train.py",
    f"--data-path {DATA_PATH}",
    "--train coco.json",
    "--val coco.json",
    "--single-cls",
    "--max-size 512",
    "--crop-size 480",
    "--model fcn_resnet50",
    "--aux-loss",
    "--loss-fn topk",
    f"--epochs {EPOCHS}",
    "-b 8",
    "-j 16",
    "--lr 0.01",
    "--print-freq 10",
    f"--output-dir /workspace/results/{EXPERIMENT_NAME}",
    f"--resume /workspace/results/{EXPERIMENT_NAME}/model_{EPOCHS-1}.pth",
    "--test-only",
])
!PYTHONPATH={CVTK_HOME} python {ARGS}
```

With dir:
```
pass
```
