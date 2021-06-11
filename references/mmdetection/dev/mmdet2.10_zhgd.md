# mmdet v2
- [mmdetection v2.10.0](https://github.com/open-mmlab/mmdetection)
- [mmcv 1.2.7](https://github.com/open-mmlab/mmcv)

## docker
- Python: 3.8
- PyTorch: 1.7.1
- MMDET: `/usr/src/mmdetection`
- [http://ip:7000/?token=hi](#) for `dev` mode
- `/usr/sbin/sshd -D -p 7000` for `ssh` mode
- `python /workspace/app_tornado.py 7000 ${@:2}` for `app` mode

```sh
docker pull flystarhe/mmdet:2.10-mmcv1.2-torch1.7-cuda11.0

n=test
t=flystarhe/mmdet:2.10-mmcv1.2-torch1.7-cuda11.0
docker run --gpus device=0 -d -p 7000:9000 --ipc=host --name ${n} -v "$(pwd)"/${n}:/workspace ${t} [dev|ssh|app]

docker update --restart=always ${n}
```

## scripts
在容器外运行程序，请先执行命令`ln -snf /root/hej/zhgd2 /workspace`完成路径映射。
```sh
%%bash

img_dir=/workspace/notebooks/xxxx
ann_dir=${img_dir}
out_dir=${img_dir}_E32
include='-i hiplot(*.csv)/coco(*.json)/dir(path/)'
mapping='{"HARD":"__DEL"}'
time python -m cvtk coco ${img_dir} -a ${ann_dir} -o ${out_dir} -m ${mapping} -e 32 --all-imgs
time python -m cvtk coco4kps 500 ${out_dir}/coco.json -n 5 --stratified
time python -m cvtk coco4lpg 1 ${out_dir}/coco.json

cp -u ${out_dir}/keep_p_samples/01/train.json ${out_dir}/coco_.json
time python -m cvtk coco4kps 0.7 ${out_dir}/coco_.json -n 3 --stratified

coco_dir=/workspace/notebooks/xxxx
coco_file=keep_p_samples/01/train.json
output_dir=${coco_dir}_VIZ
options='{"level":1,"filters":"[-]/workspace/notebooks/selected_csv,","group_by":"target"}'
time python -m cvtk viz-coco ${coco_dir} ${coco_file} ${output_dir} -o ${options}

results=/workspace/notebooks/pkl_file
mode=complex, max_score or rank_mixed
score_thr='{"*":0.3}'
label_grade='{"*":1}'
options='{"clean_mode":None,"clean_param":None}'
time python -m cvtk gen-test ${results} ${mode} ${score_thr} ${label_grade} -o ${options}

results=/workspace/notebooks/pkl_file
score_thr='{"*":0.3}'
output_dir=${results%.*}_VIZ
options='{"level":1,"filters":"[-]/workspace/notebooks/selected_csv,","group_by":"predict","clean_mode":None,"clean_param":None}'
time python -m cvtk viz-test ${results} ${score_thr} ${output_dir} -o ${options}
```

**cvtk.utils.abc.discover**
```jupyter
from cvtk.utils.abc.discover import hip_coco, hip_test, hip_test_image, hardmini_test

coco_file = ''
hip_coco(coco_file, crop_size=1280, splits=2, scales=[8], base_sizes=[4, 8, 16, 32, 64], ratios=[0.5, 1.0, 2.0])

results = ''
score_thr = {'*': 0.3}
hip_test(results, splits=2, score_thr=score_thr, match_mode='iou', min_pos_iou=0.25)

results = ''
hip_test_image(results, splits=2)

logs = ''
hardmini_test(logs, level='image', score=0.85, nok=True)
```

## parameters
```python
cfg_times = 3
cfg_lr = 0.02/4
cfg_classes = []
cfg_num_classes = 20
cfg_albu_p = 0.5
cfg_num_gpus = 2
cfg_mini_batch = 2
cfg_multi_scale = []
cfg_crop_size = 1280
cfg_load_from = None
cfg_frozen_stages = 1
cfg_experiment_path = './tmp/ipynbname'
cfg_train_data_root = '/workspace/notebooks/xxxx'
cfg_train_coco_file = 'keep_p_samples/01/train.json'
cfg_val_data_root = '/workspace/notebooks/xxxx'
cfg_val_coco_file = 'keep_p_samples/01/val.json'
cfg_test_data_root = '/workspace/notebooks/xxxx'
cfg_test_coco_file = 'keep_p_samples/01/test.json'
cfg_tmpl_path = '/usr/src/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
```

## base
执行命令`%cd /workspace/cvtk/references/mmdetection`设置工作目录。
```python
import os

#MMDET_PATH = '/usr/src/mmdetection'
#os.environ['MMDET_PATH'] = MMDET_PATH
#os.environ['MKL_THREADING_LAYER'] = 'GNU'

from cvtk.utils.notebook import clean_models
```

>`%time !jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook_ipynb`

## train
```python
%%time
albu_train_transforms = [
    dict(type='RandomRotate90', p=cfg_albu_p),
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Resize', test_mode=(len(cfg_multi_scale) == 0), multi_scale=cfg_multi_scale),
    dict(type='RandomCrop', height=cfg_crop_size, width=cfg_crop_size),
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
            dict(type='Resize', test_mode=True, multi_scale=cfg_multi_scale),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

cfg_data = dict(
    samples_per_gpu=cfg_mini_batch,
    workers_per_gpu=cfg_mini_batch,
    train=dict(
        type='CocoDataset',
        data_root=cfg_train_data_root,
        ann_file=cfg_train_coco_file,
        classes=cfg_classes,
        img_prefix='',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        data_root=cfg_val_data_root,
        ann_file=cfg_val_coco_file,
        classes=cfg_classes,
        img_prefix='',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        data_root=cfg_test_data_root,
        ann_file=cfg_test_coco_file,
        classes=cfg_classes,
        img_prefix='',
        pipeline=test_pipeline))

cfg_model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=cfg_frozen_stages,
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
            num_classes=cfg_num_classes,
        ),
    ),
)

cfg_lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-4,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
)

cfg_log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ],
)

cfg_options = dict(
    optimizer=dict(type='SGD', lr=cfg_lr, momentum=0.9, weight_decay=0.0001),
    runner=dict(type='EpochBasedRunner', max_epochs=12 * cfg_times),
    evaluation=dict(interval=6, metric='bbox'),
    checkpoint_config=dict(interval=1),
    log_config=cfg_log_config,
    lr_config=cfg_lr_config,
    load_from=cfg_load_from,
    model=cfg_model,
    data=cfg_data)
os.environ['CFG_OPTIONS'] = str(cfg_options)

ARG_TRAIN = '{} --work-dir {} --launcher pytorch'.format(cfg_tmpl_path, cfg_experiment_path)
!python -m torch.distributed.launch --nproc_per_node={cfg_num_gpus} dev/py_train.py {ARG_TRAIN}
DEL_FILES = ' '.join(clean_models(cfg_experiment_path, 3))
logs = !rm -rfv {DEL_FILES}
cfg_experiment_path
```

## test
```python
%%time
import os

work_dir = cfg_experiment_path
config = os.path.basename(cfg_tmpl_path)

data_root = cfg_test_data_root#'/workspace/notebooks/xxxx'
coco_file = cfg_test_coco_file#'keep_p_samples/01/train.json'
gpus = cfg_num_gpus
config_file = os.path.join(work_dir, config)
checkpoint_file = os.path.join(work_dir, 'latest.pth')
batch_size = 1
workers_per_gpu = 2

ARG_TEST = f'{data_root} {coco_file} {gpus} {config_file} {checkpoint_file} {batch_size} {workers_per_gpu}'
logs = !python dev/py_test.py {ARG_TEST}
print('\n'.join(logs))

from cvtk.utils.abc.discover import hardmini_test
[hardmini_test(logs, level='image', score=s, nok=True) for s in (0.3, 0.5, 0.85, 1.01)]
```

## notes
```python
cfg_model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=cfg_frozen_stages,
    ),
    neck=dict(
        num_outs=5,
        start_level=1,
        add_extra_convs='on_input',
        relu_before_extra_convs=False,
    ),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[8, 16, 32, 64, 128],
            finest_scale=56,
        ),
        bbox_head=dict(
            num_classes=cfg_num_classes,
        ),
    ),
)

cfg_lr_config = dict(
    _delete_=True,
    policy='Step',
    step=[8 * cfg_times, 11 * cfg_times],
    gamma=0.1,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
)

cfg_lr_config = dict(
    _delete_=True,
    policy='OneCycle',
    max_lr=cfg_lr,
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
    three_phase=False,
)
```

## app
start app:
```
n=test
t=flystarhe/mmdet:2.10-mmcv1.2-torch1.7-cuda10.2
docker run --gpus all -d --network=host --ipc=host --name ${n} -v "$(pwd)"/${n}:/workspace ${t} notebook

# docker exec -ti test bash
unzip -q DEMO_ZHGD_MODEL.zip -d model_drain
unzip -q DEMO_ZHGD_WEB.zip -d web_zhgd

# CUDA_VISIBLE_DEVICES=GPU_ID python web.py PORT MODEL_PATH
CUDA_VISIBLE_DEVICES=0 python /workspace/web_zhgd/web.py 7000 /workspace/model_drain
```

test app:
```python
import json
import requests

url = "http://localhost:7000/predict"
json_str = json.dumps({"image": {"color": {"path": "/workspace/model_drain/test.jpg"}}, "info": {"product_id": "123xp"}})

response = requests.post(url, data=dict(data=json_str))
if 200 == response.status_code:
    print(response.json())
else:
    print(response.text)
```
