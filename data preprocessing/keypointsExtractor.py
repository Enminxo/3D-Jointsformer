import mediapipe as mp
import numpy as np
import cv2
import os
import pandas as pd
import argparse
import tqdm
import xmltodict
import json
import pickle
import itertools
from tqdm import tqdm
import sys
import gc


def get_videoinfo(full_paths, task=''):
    """
    :param full_paths: list of paths to the images
    :param task:
    :return:
    """
    if task == 'RGB':
        video_id = ['_'.join(list(v)[0].split(os.sep)[-5:-2]) for k, v in
                    itertools.groupby(full_paths, lambda x: x.split(os.sep)[-3])]
        video_info = [list(v) for k, v in itertools.groupby(full_paths, lambda x: x.split(os.sep)[-3])]

    else:
        video_id = ['_'.join(list(v)[0].split(os.sep)[-6:-3]) for k, v in
                    itertools.groupby(full_paths, lambda x: x.split(os.sep)[-4])]
        video_info = [list(v) for k, v in itertools.groupby(full_paths, lambda x: x.split(os.sep)[-4])]

    video_dict = dict(zip(video_id, video_info))

    return video_dict


def get_Briareo_paths(root, task=''):
    """

    :param root:
    :param task: RGB or IR
    :return: path to each image file
    """

    if task == 'RGB':
        full_paths = [os.path.join(dir, f) for dir, subdirs, filenames in os.walk(root) for f in filenames if
                      os.path.splitext(f)[1] == '.png']
    else:
        full_paths = [os.path.join(dir, f) for dir, subdirs, filenames in os.walk(root)
                      if dir.split(os.sep)[-1] == 'raw' and dir.split(os.sep)[-2] == 'L'
                      for f in filenames if os.path.splitext(f)[1] == '.png']  # extension
    return full_paths


def center_crop(img, dim):
    """
    Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
    """

    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    cx, cy = int(width / 2), int(height / 2)
    cw, ch = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[cy - ch: cy + ch, cx - cw: cx + cw]
    return crop_img


def get_keypoints_ann(video_id, video_frames):
    """
    :param video_id:
    :param video_frames:
    :return:
    """
    v = 21  # number of keypoints
    c = 3  # number of dimensions of keypoint coordinates
    coord = {}
    # Mediapipe config
    # Initialize mediapipe Hands model
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)

    # todo: read video_frames
    label_id = video_id.split('_')[-2]
    total_frames = len(video_frames)
    keypoints = np.zeros((total_frames, v, c))
    for i, path_to_img in enumerate(video_frames):
        image = cv2.imread(path_to_img)
        # todo: centercrop the images if the image were too big: 480x640 for Mediapipe
        image = center_crop(image, (200, 200))
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # hand.process requires RBG image
        '''hands.process function stores the hand landmarks detection results in the variable ( results in this case)
        for each frame, the results provide a 3d landmark model for each hand detected'''
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    cx = round(landmark.x, 3)
                    cy = round(landmark.y, 3)
                    cz = round(landmark.z, 3)
                    coord[id] = [cx, cy, cz]

        data = pd.DataFrame.from_dict(coord)
        df = data.T
        result = df.to_numpy()
        if result.shape == (0, 0):
            keypoints[i] = np.zeros((v, c))
        else:
            keypoints[i] = result

    ann = {
        "total_frames": total_frames,
        "video_id": video_id,
        "category_id": label_id,
        "keypoints": keypoints.tolist(),
    }

    return ann


def to_output(video_dict: dict,
              labels_dict: dict[int, str],
              out_path: str,
              part='train'):
    annotations = []

    print('Start converting !')

    count = 0
    for k, v in tqdm(video_dict.items()):
        ann_info = get_keypoints_ann(video_id=k, video_frames=v)
        count += 1
        # print(sys.getsizeof(ann_info)) # 232Bytes
        print(count)
        annotations.append(ann_info)
        if count == 500:
            output_path = '{}/{}.pkl'.format(out_path, count)
            with open(output_path, 'wb') as fp:
                pickle.dump(annotations, fp)
            del annotations
            del count
            gc.collect()
            annotations = []
            count = 0

    print('Finished')


def main():
    parser = argparse.ArgumentParser(description='This script reads the raw images and extracts hand keypoints using '
                                                 'Mediapipe and save into numpy arrary')
    parser.add_argument('--task', type=str, default='RGB')
    parser.add_argument('--dir', type=str, default=None,
                        help='path to images files directory')
    # parser.add_argument('--labels', type=str, default=None,
    #                     help='path to label csv.')
    parser.add_argument(
        '--output', type=str, default='data/Briareo', help='path to output json file')

    parser.add_argument('--part', type=str, default='train', help='part of the dataset: train val o test')
    args = parser.parse_args()

    # todo: add loop to save train, val, and test
    # part = ['train', 'val']
    # for p in part:
    # dict of the corresponding labels str with its ids
    labels_str = ['g00', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07', 'g08', 'g09', 'g10', 'g11', 'g12_test']
    labels_ids = list(range(0, len(labels_str)))  
    labels_dict = dict(zip(labels_str, labels_ids))
    # data/Briareo/ - task
    out_path = os.path.join(args.output, args.task)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    full_paths = get_Briareo_paths(root=args.dir,
                                   task=args.task)
    video_dict = get_videoinfo(full_paths)  # returns video dict with video_id and video_frames info

    to_output(
        video_dict=video_dict,
        labels_dict=labels_dict,
        out_path=out_path,
        part=args.part)


if __name__ == '__main__':
    main()
