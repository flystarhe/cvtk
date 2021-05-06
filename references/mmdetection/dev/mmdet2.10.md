# mmdet v2
- [mmdetection v2.10.0](https://github.com/open-mmlab/mmdetection)
- [mmcv 1.2.7](https://github.com/open-mmlab/mmcv)

## docker
- Python: 3.8
- PyTorch: 1.7.1
- MMDET: `/usr/src/mmdetection`
- [http://ip:7000/?token=hi](#) for `dev`
- `/usr/sbin/sshd -D -p 7000` for `ssh` mode
- `python /workspace/app_tornado.py 7000 ${@:2}` for `app` mode

```sh
docker pull flystarhe/mmdet:2.10-mmcv1.2-torch1.7-cuda10.2

n=test
t=flystarhe/mmdet:2.10-mmcv1.2-torch1.7-cuda10.2
docker run --gpus device=0 -d -p 7000:9000 --ipc=host --name ${n} -v "$(pwd)"/${n}:/workspace ${t} [dev|ssh|app]

docker update --restart=always ${n}
```

## base
```python
import os

MMDET_PATH = '/usr/src/mmdetection'
os.environ['MMDET_PATH'] = MMDET_PATH
#os.environ['MKL_THREADING_LAYER'] = 'GNU'

def clean_models(work_dir, n=2):
    import re
    from pathlib import Path
    pattern = re.compile(r'_(\d+)')
    files = Path(work_dir).glob('model_*')

    def _num(name):
        try:
            return int(pattern.search(str(name)).group(1))
        except Exception as e:
            print(f'Warning {name} - {e}')
        return -1

    files = sorted(files, key=lambda x: _num(x))[:-n]
    return files
```

>`%time !jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to html --execute --allow-errors notebook.ipynb`

## train
```python
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

classes = None
data_root = '/workspace/datasets/xxxx'
cfg_dataset = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=os.path.join(data_root, 'keep_p_samples/0/train.json'),
        classes=classes,
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        ann_file=os.path.join(data_root, 'keep_p_samples/0/train.json'),
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=os.path.join(data_root, 'keep_p_samples/0/train.json'),
        classes=classes,
        img_prefix=data_root,
        pipeline=test_pipeline))

cfg_lr_config = dict(_delete_=True, policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[8, 11])

cfg_log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),])

cfg_options = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    runner=dict(type='EpochBasedRunner', max_epochs=12),
    evaluation=dict(interval=6, metric='bbox'),
    checkpoint_config=dict(interval=1),
    log_config=cfg_log_config,
    lr_config=cfg_lr_config,
    data=cfg_dataset)
os.environ['CFG_OPTIONS'] = str(cfg_options)

WORK_DIR = '/workspace/results/{}/{}'.format('task_name', 'lr_1x_epochs_1x')
CONFIG = os.path.join(MMDET_PATH, 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

ARG_TRAIN = '{} --work-dir {} --launcher pytorch'.format(CONFIG, WORK_DIR)
!python -m torch.distributed.launch --nproc_per_node=2 dev/py_train.py {ARG_TRAIN}
DEL_FILES = ' '.join(clean_models(WORK_DIR, 2))
logs = !rm -rfv {DEL_FILES}
```