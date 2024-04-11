import mediapipe
import pandas as pd
from mediapipe import solutions
import cv2
import numpy as np
from tools import test
from tools.test import my_inference
import argparse
import os
import os.path as osp
import warnings
from sklearn.metrics import confusion_matrix
import mmcv
import torch
from mmcv import Config
import source  # noqa
from mmaction.apis import inference_recognizer, init_recognizer

config_dir = 'configs/tscnn/tscnn_ntu60_xsub_joint.py'
# '/home/enz/code/skelact/configs/tscnn/tscnn_ntu60_xsub_joint.py'
cfg = Config.fromfile(config_dir)
checkpoint_file = 'work_dirs/tscnn_ntu60_xsub_joint_temp/best_top1_acc_epoch_25.pth'
# '/home/enz/code/skelact/work_dirs/tscnn_ntu60_xsub_joint_temp/best_top1_acc_epoch_175.pth'

img_path = './temp_img'
if not os.path.exists(img_path):
    os.makedirs(img_path)

coord_path = './temp_coord'
if not os.path.exists(coord_path):
    os.makedirs(coord_path)

# Mediapipe
mp_hands = solutions.hands
mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

cap = cv2.VideoCapture(0)
count = 0
frames = 0
max_frames = 32
num_joints = 21
num_hands = 1
all_data = np.zeros((1, num_hands, max_frames, 21, 3), dtype=np.float32)
input_data = dict()
""" confidence detection: threshold for the initial detection to be successful
tracking confidence: threshold for tracking after the initial detection"""
# hands = var where the landmarks points are stored

while cap.isOpened():
    ret, screen = cap.read()
    # BGR 2 RGB
    image = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    # Detections
    results = hands.process(image)
    # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        # print('--- detecting hands ---')
        if frames <= max_frames:
            for hand_landmarks in results.multi_hand_landmarks:  # for each list of normalized landmarks
                # print(hand_landmarks)
                dic = {}
                for id, landmark in enumerate(hand_landmarks.landmark):
                    # print(id, landmark)
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=1),
                        mp_drawing_styles.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=1))
                    cx = round(landmark.x, 2)
                    cy = round(landmark.y, 2)
                    # cz = round(landmark.z, 2)
                    dic[id] = [cx, cy, 0]
                data = np.array(list(dic.values()))
                # print('-', frames)
                all_data[:, :, frames, :, :] = data  # m, t, v, c
                cv2.imshow('Hand Tracking', image)
                cv2.imwrite(img_path + "/" + str(frames) + '.png', image)
                df = pd.DataFrame(data)
                df.to_csv(coord_path + "/" + str(frames) + '.csv')
        frames = frames + 1
        rep_image = image
        # print('--- ', frames)

        if frames % 32 == 0:
            # evaluate the coord with the pretrained model in # n, m, t, v, c
            input_data['keypoint'] = torch.tensor(np.transpose(all_data, (0, 4, 2, 3, 1)))
            input_data['label'] = 0
            outputs = my_inference(cfg, checkpoint_file, data_loader=input_data)
            probs = outputs[0]
            y_pred = np.argmax(probs, axis=-1)
            classes = ['C', 'down', 'fist_moved', 'five', 'four', 'hang', 'heavy', 'index', 'L', 'ok', 'palm',
                           'palm_m', 'palm_u', 'three', 'two', 'up']
            idx = np.linspace(0, len(classes), len(classes), False).astype(int)
            classes_dic = dict(zip(idx, classes))
            class_str = classes_dic.get(y_pred)
            print(class_str)
            cv2.putText(rep_image, class_str, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('recognize gesture', rep_image)
            # reset the frames
            frames = 0

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
