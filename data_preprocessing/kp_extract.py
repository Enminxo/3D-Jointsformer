import sys
import mediapipe as mp
from mediapipe import solutions

import numpy as np
import matplotlib as plt
import numpy as np
import cv2
import os
import pandas as pd
import argparse
import tqdm
import xmltodict
import json, pickle
import itertools
from tqdm import tqdm
import cProfile
import gc


def get_videoinfo(full_paths, task=''):
    """
    :param full_paths: list of paths to the images
    :param task:
    :return: video id and video info ( repetition)
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
        full_paths = [os.path.join(dir, f) for dir, subdirs, filenames in os.walk(root)
                      if dir.split(os.sep)[-3] != 'g12_test'
                      for f in filenames if os.path.splitext(f)[1] == '.png']
    else:
        full_paths = [os.path.join(dir, f) for dir, subdirs, filenames in os.walk(root)
                      if dir.split(os.sep)[-1] == 'raw' and dir.split(os.sep)[-2] == 'L' and dir.split(os.sep)[
                          -4] != 'g12_test'
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


def gen_ann(video_id, video_frames, out_path):
    """
    extract hand_pose at frame level
    :param path_to_img:
    :return: hand pose keypoints of a single video
    """

    ann = dict()
    v = 21  # num of keypoints
    c = 3  # dimensions of keypoint coordinates
    # Initialize mediapipe Hands model
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_draw_style = mp.solutions.drawing_utils
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5)
    total_frames = len(video_frames)
    kp = np.zeros((total_frames, v, c), dtype=np.float32)
    for i, path_to_img in enumerate(video_frames):
        image = cv2.imread(path_to_img)
        # todo: centercrop the images if the image were too big: 480x640 for Mediapipe
        # image = center_crop(image, (200, 200))
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # hand.process requires RBG image
        hand_landmarks_list = results.multi_hand_landmarks
        """ hands.process returns the hand landmarks detection results
        for each frame, the results provide a 3d landmark model for each hand detected
        landmark = normalizedLandmark ; hand_landmark = NormalizedLandmarkList"""
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                coord_list = []
                for landmark in hand_landmarks.landmark:
                    # todo: en cada iteracion guarda coord de 1 keypoint
                    coord_list.append(np.around(np.array([landmark.x, landmark.y, landmark.z]), 3))
                    mp_draw.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_style.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                        mp_draw_style.DrawingSpec(color=(250, 44, 250), thickness=1, circle_radius=1))
                    # cv2.imwrite('{}/{}_{}.png'.format(out_path, video_id, str(i)), image)
                hand_pose = np.array(coord_list)
                if hand_pose.shape != (0, 0):
                    kp[i] = hand_pose
                else:
                    kp[i] = np.zeros((v, c))  # frame level hand pose
    # video level
    ann["video_id"] = video_id
    ann["label"] = video_id.split('_')[-2]
    ann["total_frames"] = kp.shape[0]
    ann["frames"] = video_frames
    ann["keypoint"] = kp[..., :]

    output_path = '{}/{}.pkl'.format(out_path, video_id)
    with open(output_path, 'wb') as fp:
        pickle.dump(ann, fp)


def parse_args():
    parser = argparse.ArgumentParser(description='This script reads the raw images and extracts hand keypoints using '
                                                 'Mediapipe and Generate the annotation file for a single video')
    parser.add_argument('--task', type=str, default='RGB', help=' RGB or IR')
    parser.add_argument('--dir', type=str, default=None,
                        help='path to images files directory')

    parser.add_argument(
        '--output', type=str, default='data/briareo', help='path to output ann file')

    parser.add_argument('--part', type=str, default='train', help='part of the dataset: train val o test')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    task_pth = os.path.join(args.output, args.task)
    os.makedirs(task_pth, exist_ok=True)

    out_path = os.path.join(task_pth, args.part)
    os.makedirs(out_path, exist_ok=True)

    full_paths = get_Briareo_paths(root=args.dir, task=args.task)

    video_dict = get_videoinfo(full_paths, task=args.task)  # returns video dict with video_id and video_frames info
    # for k, v in tqdm(video_dict.items()):
    #     gen_ann(k, v, out_path)

    # todo: split the dict in 2 for train
    d1 = dict(list(video_dict.items())[:len(video_dict) // 2])
    # for k, v in tqdm(d1.items()):
    #     gen_ann(k, v, out_path)

    d2 = dict(list(video_dict.items())[len(video_dict) // 2:])
    for k, v in tqdm(d2.items()):
        gen_ann(k, v, out_path)
