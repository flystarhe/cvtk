# mmdet v2
- [mmdetection v2.11.0](https://github.com/open-mmlab/mmdetection)
- [mmcv 1.3.3](https://github.com/open-mmlab/mmcv)

## docker
- Python: 3.8
- PyTorch: 1.7.1
- MMDET: `/usr/src/mmdetection`
- [http://ip:7000/?token=hi](#) for `dev`
- `/usr/sbin/sshd -D -p 7000` for `ssh` mode
- `python /workspace/app_tornado.py 7000 ${@:2}` for `app` mode

```sh
docker pull flystarhe/mmdet:2.11-mmcv1.3-torch1.7-cuda11.0

n=test
t=flystarhe/mmdet:2.11-mmcv1.3-torch1.7-cuda11.0
docker run --gpus device=0 -d -p 7000:9000 --ipc=host --name ${n} -v "$(pwd)"/${n}:/workspace ${t} [dev|ssh|app]

docker update --restart=always ${n}
```

## data
```python
img_dir=dataset
ann_dir=${img_dir}
out_dir=${img_dir}_coco
include='-i hiplot(*.csv)/coco(*.json)/dir(path/)'
mapping='{"HARD":"__DEL"}'
python -m cvtk.utils.abc coco ${img_dir} -a ${ann_dir} -o ${out_dir} -m ${mapping} -e 32
python -m cvtk.utils.abc coco4kps 500 ${out_dir}/coco.json --stratified

results=/workspace/results/pkl_file
score_thr='{"*":0.3}'
output_dir=/workspace/results/pkl_file_stem
options='{"include":"/workspace/results/selected_csv"}'
python -m cvtk.utils.abc viz-test ${results} ${score_thr} ${output_dir} -o ${options}
```

## base
```python
%cd /workspace/cvtk/references/mmdetection
import os

MMDET_PATH = '/usr/src/mmdetection'
os.environ['MMDET_PATH'] = MMDET_PATH
#os.environ['MKL_THREADING_LAYER'] = 'GNU'

def clean_models(work_dir, n=2):
    import re
    from pathlib import Path
    pattern = re.compile(r'_(\d+)')
    files = Path(work_dir).glob('epoch_*')

    def _num(name):
        try:
            return int(pattern.search(name).group(1))
        except Exception as e:
            print(f'Warning {name} - {e}')
        return -1

    files = [str(f) for f in files]
    files = sorted(files, key=lambda x: _num(x))[:-n]
    return files
```

>`%time !jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook.ipynb`

## train
```python
%%time
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', test_mode=True, force_square=False),
    dict(type='RandomCrop', height=800, width=800),
    dict(type='RandomFlip', flip_ratio=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        transforms=[
            dict(type='Resize', test_mode=True, force_square=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

xlr = 1
times = 1
classes = None
num_classes = 20
data_root = '/workspace/datasets/xxxx'
coco_file = 'keep_p_samples/01/train.json'
group = 'task_name'
project = f'lr_{xlr}x_epochs_{times}x'

cfg_data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=os.path.join(data_root, coco_file),
        classes=classes,
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file=os.path.join(data_root, coco_file),
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=os.path.join(data_root, coco_file),
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline))

cfg_model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
        ),
    ),
)

cfg_lr_config = dict(
    _delete_=True,
    policy='Step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8 * times, 11 * times],
)

cfg_log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

cfg_options = dict(
    optimizer=dict(type='SGD', lr=0.01 * xlr, momentum=0.9, weight_decay=0.0001),
    runner=dict(type='EpochBasedRunner', max_epochs=12 * times),
    evaluation=dict(interval=2, metric='bbox'),
    checkpoint_config=dict(interval=1),
    log_config=cfg_log_config,
    lr_config=cfg_lr_config,
    model=cfg_model,
    data=cfg_data)
os.environ['CFG_OPTIONS'] = str(cfg_options)

WORK_DIR = '/workspace/results/{}/{}'.format(group, project)
CONFIG = os.path.join(MMDET_PATH, 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

ARG_TRAIN = '{} --work-dir {} --launcher pytorch'.format(CONFIG, WORK_DIR)
!python -m torch.distributed.launch --nproc_per_node=2 dev/py_train.py {ARG_TRAIN}
DEL_FILES = ' '.join(clean_models(WORK_DIR, 2))
logs = !rm -rfv {DEL_FILES}
f'WORK_DIR: {WORK_DIR}'
```

## test
```python
%%time
import os

times = 1
work_dir = WORK_DIR
config = os.path.basename(CONFIG)
data_root = '/workspace/datasets/xxxx'
coco_file = 'keep_p_samples/01/train.json'

gpus = 2
config_file = os.path.join(work_dir, config)
checkpoint_file = os.path.join(work_dir, f'epoch_{12 * times}.pth')
batch_size = 1
workers_per_gpu = 2

ARG_TEST = f'{data_root} {coco_file} {gpus} {config_file} {checkpoint_file} {batch_size} {workers_per_gpu}'
!python dev/py_test.py {ARG_TEST}
```

## notes
```python
cfg_model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
        ),
    ),
)

cfg_lr_config = dict(
    _delete_=True,
    policy='Step',
    gamma=0.1,
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=8 * times,
)

cfg_lr_config = dict(
    _delete_=True,
    policy='OneCycle',
    max_lr=0.01 * xlr,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
    three_phase=False,
)

from cvtk.utils.abc.discover import hip_coco, hip_test, hip_test_image

coco_file = ''
hip_coco(coco_file, crop_size=800, splits=2, scales=[8], base_sizes=[4, 8, 16, 32, 64], ratios=[0.5, 1.0, 2.0])

results = ''
score_thr = {"*": 0.3}
hip_test(results, splits=2, score_thr=score_thr, clean_mode="min", clean_param=0.1, match_mode="iou", min_pos_iou=0.25)

results = ''
mode = None
score_thr = {"*": 0.3}
label_grade = {"*": 1}
kw = {}
hip_test_image(results, splits=2, mode=mode, score_thr=score_thr, label_grade=label_grade, **kw)
```
