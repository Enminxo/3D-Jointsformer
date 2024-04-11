# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pickle
import time
import sklearn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import os
import os.path as osp
import warnings
import numpy as np
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)
import source  # noqa
from plot_results import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        outputs = single_gpu_test(model, data_loader)
    else:
        model = build_ddp(
            model,
            default_device,
            default_args=dict(
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False))
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--result', type=str, default='output/briareo', help='path to save the output files')
    parser.add_argument('--task', type=str, default='RGB', help=' RGB or IR')
    parser.add_argument('--title', type=str, help=' title of the confusion matrix')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
             ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. For example, '
             "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    # todo: make dirs
    os.makedirs(args.result, exist_ok=True)
    task_pth = os.path.join(args.result, args.task)
    os.makedirs(task_pth, exist_ok=True)

    if args.tensorrt and args.onnx:
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        if 'output_format' in output_config:
            # ugly workround to make recognition and localization the same
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            out = output_config['out']
            # make sure the dirname of the output path exists
            mmcv.mkdir_or_exist(osp.dirname(out))
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    # dataset.video_infos -> list
    # todo: get the true labels
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    lst_dic = [i for i in dataset.video_infos]
    y_true = [value for d in lst_dic for (key, value) in d.items() if key == 'label']
    out_path = args.result
    # output_path = '{}/{}.pkl'.format(out_path, 'briareo_y_true')
    output_path = '{}/{}.pkl'.format(out_path, 'gti_y_true')
    with open(output_path, 'wb') as fp:
        pickle.dump(y_true, fp)

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # if args.tensorrt:
    #     outputs = inference_tensorrt(args.checkpoint, distributed, data_loader,
    #                                  dataloader_setting['videos_per_gpu'])
    # elif args.onnx:
    #     outputs = inference_onnx(args.checkpoint, distributed, data_loader,
    #                              dataloader_setting['videos_per_gpu'])
    # else:
    #     outputs = inference_pytorch(args, cfg, distributed, data_loader)
    start_inference = time.time()
    outputs = inference_pytorch(args, cfg, distributed, data_loader)
    finish_inference = time.time()
    inference_time = finish_inference - start_inference
    print("inference time:", inference_time)
    # todo: plot confusion matrix
    y_pred = np.argmax(outputs, axis=-1)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1score = f1_score(y_true, y_pred, average='micro')
    # print("\n")
    # print(f"Precision = {precision}")
    # print(f"Recall = {recall}")
    # print(f"F1 Score = {f1score}")
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    briareo_labels = ['fist', 'pinch', 'flip-over', 'telephone', 'right swipe', 'left swipe', 'top-down swipe',
                      'bottom-up swipe', 'thumb', 'index', 'clockwise rotation', 'counterclockwise rotation']
    gti_labels = ['C', 'down', 'fist_moved', 'five', 'four', 'hang', 'heavy', 'index', 'L', 'ok', 'palm', 'palm_m',
                  'palm_u', 'three', 'two', 'up']
    # todo: change the labels
    # plot_confusion_matrix(path=task_pth, cm=cm, classes=briareo_labels, title=args.title)
    # plot_confusion_matrix(path=task_pth, cm=cm, classes=gti_labels, title=args.title)

    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None):
            out = output_config['out']
            print(f'\nwriting results to {out}')
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("main ", time.time() - start_time)
