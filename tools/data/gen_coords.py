# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import random
import os
import os.path as osp
import pandas as pd
import mmcv
import numpy as np


def read_xy(path, file, num_joint=21):
    # file = csv file name
    df = pd.read_csv(os.path.join(path, file))  # dataframe
    seq_info = df.iloc[:, [1, 2]]  # columns x y
    data = np.zeros((num_joint, 3))
    for idx, row in seq_info.iterrows():
        data[idx, :] = [row['0'], row['1'], 0]  # write coordinates
    return data


def prueba(data_path, out_path, task='ntu60', benchmark='xsub', part='train'):
    if benchmark == 'xsub':
        if task == 'ntu60':
            results = []
            num_joint = 21
            maxhand = 1
            n_frames = 32
            classes = ['C', 'down', 'fist_moved', 'five', 'four', 'hang', 'heavy', 'index', 'L', 'ok', 'palm', 'palm_m',
                       'palm_u', 'three', 'two', 'up']
            idx = np.linspace(0, len(classes), len(classes), False).astype(int)
            classes_dic = dict(zip(classes, idx))
            for path, _, filename in os.walk(data_path):  # filename = list of filenames of each class
                label = path.split(os.sep)[-1]
                label_idx = classes_dic.get(label)
                total_frames = len(filename)
                fp = np.zeros((len(classes), maxhand, total_frames, num_joint, 3), dtype=np.float32)
                # print(total_frames)
                if total_frames != 0:
                    count = total_frames // n_frames
                    for i, f in enumerate(filename):
                        # print(label, f)  # C frame_132423_r.png.csv
                        data = read_xy(os.path.join(data_path, label), f, num_joint=21).astype(np.float32)
                        fp[label_idx, :, i, :, :] = data
                        # mega matriz donde se guardan todos los frames de la misma clase
                    for c in range(count): # save every 32 frames
                        anno = dict()
                        anno['total_frames'] = 32
                        ant = c * n_frames
                        # anno['keypoint'] = fp[label_idx, :, ant: (c + 1) * n_frames, :, :]  # M T V C
                        anno['imgs'] = fp[label_idx, :, ant: (c + 1) * n_frames, :, :]  # M T V C
                        anno['label'] = label_idx
                        anno['name'] = filename[ant: (c + 1) * n_frames]  # list of filenames contained in a class
                        results.append(anno)  # repite lo mismo "count" veces
                        # print(c, ant, (c + 1) * n_frames)
                    # print(label_idx)
            print('finished')
            output_path = '{}/{}.pkl'.format(out_path, part)
            mmcv.dump(results, output_path)

    return

'''
def another_prueba(data_path, out_path, task='ntu60', benchmark='xsub', part='val'):
    if benchmark == 'xsub':
        if task == 'ntu60':
            results = []
            num_joint = 21
            maxhand = 1
            classes = ['C', 'down', 'fist_moved', 'five', 'four', 'hang', 'heavy', 'index', 'L', 'ok', 'palm', 'palm_m',
                       'palm_u', 'three', 'two', 'up']
            idx = np.linspace(0, len(classes), len(classes), False).astype(int)
            classes_dic = dict(zip(classes, idx))
            for path, _, filename in os.walk(data_path):  # filename = list of filenames of each class
                label = path.split(os.sep)[-1]
                label_idx = classes_dic.get(label)
                total_frames = len(filename)
                fp = np.zeros((len(classes), maxhand, total_frames, num_joint, 3), dtype=np.float32)
                if total_frames != 0:  # saltar el primer bucle xq la lista está vacía
                    for i, f in enumerate(filename):
                        # print(label, f)  # C frame_132423_r.png.csv
                        data = read_xy(os.path.join(data_path, label), f, num_joint=21).astype(np.float32)
                        fp[label_idx, :, i, :, :] = data
                    anno = dict()
                    anno['total_frames'] = len(filename)
                    anno['keypoint'] = fp[label_idx, :, 0:len(filename), :, :]  # M T V C
                    anno['label'] = label_idx
                    anno['name'] = filename  # list of filenames contained in a class
                    results.append(anno)
                    print(label_idx)
            # re = results[1:]
            # print('finished')
            # output_path = '{}/{}.pkl'.format(out_path, part)
            # mmcv.dump(results[1:], output_path)

    return

'''
'''
def mygendata(data_path, out_path, task='ntu60', benchmark='xsub', part='val'):
    if benchmark == 'xsub':
        if task == 'ntu60':
            results = []
            total_frames = []
            total = []
            sample_label = ['C', 'down', 'fist_moved', 'five', 'four', 'hang', 'heavy', 'index', 'L', 'ok', 'palm',
                            'palm_m', 'palm_u', 'three', 'two', 'up']
            max_frames = 6000
            num_joint = 21
            maxhand = 1
            fp = np.zeros((len(sample_label), maxhand, max_frames, num_joint, 3),
                          dtype=np.float32)  #

            for i, c in enumerate(sample_label):
                dic = {}  # 16
                frames = os.listdir(os.path.join(data_path, c))  # list of frames belonging to the same class
                # print(len(frames))
                # total_frames.append(frames)  # list of lists
                total_frames.append(len(frames))
                dic[c] = frames
                total.append(dic)  # list of 16 dictionaries with key = label, and value = list of frames

            for e, d in enumerate(total):
                for label, lst_f in d.items():
                    for j, f in enumerate(lst_f):  # f = file_name
                        data = read_xy(
                            os.path.join(data_path, label),
                            f,
                            num_joint=21).astype(np.float32)
                        fp[e, :, j, :, :] = data
                        anno = dict()
                        anno['total_frames'] = len(lst_f)  # 226,213,163...
                        # M T V C
                        anno['keypoint'] = fp[e, :, 0: len(lst_f), :, :]
                        anno['label'] = e
                        anno['filename'] = lst_f

                        # for frame in total_frames:  # 16
                        #     anno = dict()
                        #     anno['total_frames'] = frame  # 226,213,163...
                        #     # M T V C
                        #     anno['keypoint'] = fp[e, :, 0:frame, :, :]
                        #     anno['label'] = e
                        # anno['filenames'] =
                        results.append(anno)
            print('finished')
            output_path = '{}/{}.pkl'.format(out_path, part)
            mmcv.dump(results, output_path)

    return

'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for NTURGB-D raw skeleton data')
    parser.add_argument(
        '--data-path',
        type=str,
        help='raw skeleton data path',
        default='data/ntu/nturgb+d_skeletons_60/')
    parser.add_argument(
        '--ignored-sample-path',
        type=str,
        default='NTU_RGBD_samples_with_missing_skeletons.txt')
    parser.add_argument(
        '--out-folder', type=str, default='data/ntu/nturgb+d_skeletons_60_3d')
    parser.add_argument('--task', type=str, default='ntu60')
    parser.add_argument('--benchmark', type=str, default='xsub')
    parser.add_argument('--part', type=str, default='train')
    args = parser.parse_args()

    assert args.task in ['ntu60', 'ntu120']

    out_path = osp.join(args.out_folder, args.benchmark)
    if not osp.exists(out_path):
        os.makedirs(out_path)
    prueba(data_path=args.data_path,
           out_path=out_path,
           task=args.task,
           benchmark=args.benchmark,
           part=args.part)

"""
Args: 
part:  either train or val
data_path: train_coord or test_coord

"""