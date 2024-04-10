import numpy as np
import os
import sklearn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import pickle
import numpy as np
import torch
from torch import nn
import torch.onnx
import pandas as pd
import matplotlib.pyplot as plt



# path = './data/ntu_prueba/nturgb+d_skeletons_60_3d/xsub/train.pkl'
# path = './data/briareo/xsub/train.pkl'

# with open(path, 'rb') as pickle_file:
#     content = pickle.load(pickle_file)

# dataframe = pd.read_pickle(path)  # list of dictionaries

#
# with open("output/briareo/SlowOnly_2LayersTransformer.pkl", 'rb') as f:
#    briareo_preds = pickle.load(f)
#
# with open("output/gti/SlowOnly_2LayersTransformer_gti.pkl", 'rb') as f:
#     gti_preds= pickle.load(f)


def plot(path):
    dataframe = pd.read_pickle(path)  # list of dictionaries
    for i, data in enumerate(dataframe[10:30]):
        label = dataframe[i]['label']
        df = dataframe[i]['keypoint']  # ndarrary 1,32,21,3
        df = df.squeeze(axis=0)
        for f in range(32):
            sample = df[f, :, :]
            # Todo: normalización respecto de los pixels min y max
            # prueba_x = (sample[:,0] - sample[:,0].min()) / (sample[:,0].max() - sample[:,0].min())
            # prueba_y = (sample[:,1] - sample[:,1].min()) / (sample[:,1].max() - sample[:,1].min())
            # Todo: normalización respecto de un nodo
            prueba_x = np.abs((sample[:, 0] - sample[0, 0])) / sample[0, 0]
            prueba_y = np.abs((sample[:, 1] - sample[0, 1])) / sample[0, 1]
            # prueba = df[f, :, :]
            # plt.scatter(prueba_x, 1 - prueba_y)
            plt.plot(prueba_x, prueba_y)
            plt.title(str(label))
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            plt.show()

# plot(path=path)
