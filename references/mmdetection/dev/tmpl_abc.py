from cvtk.utils import notebook

cfg = dict(
    cfg_times=3,
    cfg_lr=0.02/4,
    cfg_classes=[],  # must modify
    cfg_num_classes=20,  # must modify
    cfg_albu_p=0.5,  # must modify
    cfg_num_gpus=2,
    cfg_mini_batch=2,
    cfg_multi_scale=[],
    cfg_crop_size=1280,
    cfg_load_from=None,
    cfg_frozen_stages=1,
    cfg_experiment_path='./tmp/ipynbname',
    cfg_train_data_root='/workspace/notebooks/xxxx',  # must modify
    cfg_train_coco_file='keep_p_samples/01/train.json',  # must modify
    cfg_val_data_root='/workspace/notebooks/xxxx',  # must modify
    cfg_val_coco_file='keep_p_samples/01/val.json',  # must modify
    cfg_test_data_root='/workspace/notebooks/xxxx',  # must modify
    cfg_test_coco_file='keep_p_samples/01/test.json',  # must modify
    cfg_tmpl_path='/usr/src/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
)

workdir = '/workspace/cvtk/references/mmdetection'
outdir = cfg['cfg_train_data_root'] + '_T0000'
# !rm -rf {outdir}

task_list = [
    dict(cfg_times=2, cfg_lr=0.02/16*4, cfg_num_gpus=2, cfg_mini_batch=2,
         cfg_multi_scale=[], cfg_crop_size=800, group_id=1,
         cfg_train_coco_file='keep_p_samples/01/train.json',
         cfg_test_coco_file='keep_p_samples/01/test.json',
         cfg_val_coco_file='keep_p_samples/01/test.json',
         input_path='/workspace/cvtk/references/mmdetection/dev/tmpl_s1_cos.ipynb'),
    dict(cfg_times=2, cfg_lr=0.02/16*4, cfg_num_gpus=2, cfg_mini_batch=2,
         cfg_multi_scale=[], cfg_crop_size=800, group_id=1,
         cfg_train_coco_file=[f'keep_p_samples/{i:02d}/train.json' for i in range(1, 5)],
         cfg_test_coco_file='keep_p_samples/01/all.json',
         cfg_val_coco_file='keep_p_samples/01/all.json',
         input_path='/workspace/cvtk/references/mmdetection/dev/tmpl_s1_cos.ipynb'),
    dict(cfg_times=1, cfg_lr=0.02/16*4/25, cfg_num_gpus=2, cfg_mini_batch=2,
         cfg_multi_scale=[], cfg_crop_size=0, group_id=1,
         cfg_train_coco_file='keep_p_samples/01/all.json',
         cfg_test_coco_file='keep_p_samples/01/all.json',
         cfg_val_coco_file='keep_p_samples/01/all.json',
         input_path='/workspace/cvtk/references/mmdetection/dev/tmpl_s1_cos.ipynb'),
]
carry_on = None  # None, 'latest', ['key-name']
res1, cached1 = notebook.run(task_list, carry_on, workdir, outdir, cfg)
print(cached1)

nbs = ' '.join([log['metadata']['papermill']['output_path'] for log in res1])
# !jupyter nbconvert --to html {nbs}
# !rm -rfv {nbs}
