import copy
from pathlib import Path
from cvtk.text.utils import replace2
from papermill import execute_notebook

workdir = '/workspace/cvtk/references/mmdetection'
_cfg = dict(cfg_xlr=1.0,
            cfg_times=3,
            cfg_classes=[],
            cfg_num_classes=20,
            cfg_albu_p=0.5,
            cfg_experiment_path='./tmp/ipynbname',
            cfg_train_data_root='/workspace/notebooks/xxxx',
            cfg_train_coco_file='keep_p_samples/01/train.json',
            cfg_val_data_root='/workspace/notebooks/xxxx',
            cfg_val_coco_file='keep_p_samples/01/test.json',
            cfg_test_data_root='/workspace/notebooks/xxxx',
            cfg_test_coco_file='keep_p_samples/01/test.json',
            cfg_tmpl_path='/usr/src/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
outdir = Path(_cfg['cfg_train_data_root'] + '_T0601a')
# !rm -rf {outdir} && mkdir -p {outdir}

logs = []
for input_path in ['test.ipynb']:
    for gid in [1, 2, 3]:
        for xlr in [1.0, 2.0, 0.5]:
            cfg = copy.deepcopy(_cfg)
            input_path = Path(input_path)
            fn = f'{input_path.stem}_{gid=:02d}_{xlr=:.1f}.ipynb'
            cfg = replace2(r'.*coco_file$', cfg, '/01/', f'/{gid:02d}/')
            cfg.update(cfg_xlr=xlr, cfg_experiment_path=str(outdir / fn[:-6]))
            log = execute_notebook(input_path, outdir / fn, parameters=cfg,
                                   cwd=workdir, progress_bar=False,
                                   kernel_name='python')
            logs.append(log)

nbs = ' '.join([log['metadata']['papermill']['output_path'] for log in logs])
# !jupyter nbconvert --to html {nbs}
# !rm -rfv {nbs}
