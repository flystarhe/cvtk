from cvtk.utils import notebook

cfg = dict(
    cfg_times=3,
    cfg_lr=0.02/4,
    cfg_classes=[],
    cfg_num_classes=20,
    cfg_albu_p=0.5,
    cfg_num_gpus=2,
    cfg_mini_batch=2,
    cfg_multi_scale=[],
    cfg_crop_size=1280,
    cfg_load_from=None,
    cfg_experiment_path='./tmp/ipynbname',
    cfg_train_data_root='/workspace/notebooks/xxxx',
    cfg_train_coco_file='keep_p_samples/01/train.json',
    cfg_val_data_root='/workspace/notebooks/xxxx',
    cfg_val_coco_file='keep_p_samples/01/test.json',
    cfg_test_data_root='/workspace/notebooks/xxxx',
    cfg_test_coco_file='keep_p_samples/01/test.json',
    cfg_tmpl_path='/usr/src/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
)
workdir = '/workspace/cvtk/references/mmdetection'
outdir = cfg['cfg_train_data_root'] + '_T0000'
# !rm -rf {outdir} && mkdir -p {outdir}

carry_on = None  # 'latest', 0, [0, 2]
input_path_list = ['test.ipynb']
times_list = [2, 4, 6]
lr_list = [0.02/4]
gids = [1]
res1 = notebook.run(carry_on, workdir, outdir, cfg, input_path_list,
                    times_list, lr_list, gids)

nbs = ' '.join([log['metadata']['papermill']['output_path'] for log in res1])
# !jupyter nbconvert --to html {nbs}
# !rm -rfv {nbs}
