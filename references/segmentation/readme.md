# Semantic segmentation reference training scripts
Tested on `pytorch:1.7.1`.

## docker
* python: 3.8
* pytorch: 1.7.1

```
docker pull registry.cn-hangzhou.aliyuncs.com/flystarhe/containers:torch1.7.1-dev
docker tag registry.cn-hangzhou.aliyuncs.com/flystarhe/containers:torch1.7.1-dev torch:1.7.1-dev

docker save -o torch1.7.1-dev-21.03.tar torch:1.7.1-dev
docker load -i torch1.7.1-dev-21.03.tar

t=test && docker run --gpus all -d -p 9000:9000 --ipc=host --name ${t} -v "$(pwd)"/${t}:/workspace torch:1.7.1-dev
```

## training scripts
Assume `git clone https://github.com/flystarhe/cvtk.git /workspace/cvtk`. you must modify the following flags:

* `--nproc_per_node=<number_of_gpus_available>`

```
import os
import time

CVTK_HOME = "/workspace/cvtk"
!cd {CVTK_HOME} && git log -1 --oneline
os.environ["MKL_THREADING_LAYER"] = "GNU"
EXPERIMENT_NAME = time.strftime("T%m%d_%H%M")

DATA_PATH = "/workspace/coco"
ARGS = " ".join([
    "-m torch.distributed.launch --nproc_per_node=2 --use_env",
    f"{CVTK_HOME}/references/segmentation/train.py",
    f"--data-path {DATA_PATH}",
    "--train coco.json",
    "--val coco.json",
    "--single-cls",
    "--crop-size 480",
    "--model fcn_resnet50",
    "--aux-loss",
    "--epochs 30",
    "-b 8",
    "-j 16",
    "--lr 0.02",
    "--print-freq 10",
    f"--output-dir /workspace/results/{EXPERIMENT_NAME}",
])
!PYTHONPATH={CVTK_HOME} python {ARGS}
```

## test scripts
```
import os
import time

CVTK_HOME = "/workspace/cvtk"
!cd {CVTK_HOME} && git log -1 --oneline
os.environ["MKL_THREADING_LAYER"] = "GNU"
EXPERIMENT_NAME = time.strftime("T%m%d_%H%M")

DATA_PATH = "/workspace/coco"
CHECKPOINT = "/workspace/results/xxxx/model_30.pth"
ARGS = " ".join([
    "-m torch.distributed.launch --nproc_per_node=2 --use_env",
    f"{CVTK_HOME}/references/segmentation/train.py",
    f"--data-path {DATA_PATH}",
    "--train coco.json",
    "--val coco.json",
    "--single-cls",
    "--crop-size 480",
    "--model fcn_resnet50",
    "--aux-loss",
    "--epochs 30",
    "-b 8",
    "-j 16",
    "--lr 0.02",
    "--print-freq 10",
    f"--output-dir /workspace/results/{EXPERIMENT_NAME}",
    f"--resume {CHECKPOINT}",
    "--test-only",
])
!PYTHONPATH={CVTK_HOME} python {ARGS}
```

With dir:
```
pass
```
