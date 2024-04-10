import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

"""create custom dataset by making our dataset a subclass of Pytorch Dataset"""


class normalize_coord(torch.nn.Module):
    """max-min normalization """

    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        # sample is a tensor of keypoints -> 1,32,21,3
        # normalizar a nivel de frame
        sample = sample.numpy()
        final = np.zeros((1, 32, 21, 3))
        for h in range(sample.shape[0]):
            for f in range(sample.shape[1]):
                # Todo: normalización respecto de los pixels min y max
                # pixels_x = (sample[h, f, :, 0] - sample[h, f, :, 0].min()) / (
                #         sample[h, f, :, 0].max() - sample[h, f, :, 0].min())
                # pixels_y = (sample[h, f, :, 1] - sample[h, f, :, 1].min()) / (
                #             sample[h, f, :, 1].max() - sample[h, f, :, 1].min())

                # Todo: normalización respecto de un nodo
                pixels_x = np.abs(sample[h, f, :, 0] - sample[h, f, 0, 0]) / sample[h, f, 0, 0]
                pixels_y = np.abs(sample[h, f, :, 1] - sample[h, f, 0, 1]) / sample[h, f, 0, 0]

                final[h, f, :, 0] = pixels_x
                final[h, f, :, 1] = pixels_y

        return torch.tensor(final)


class center_crop(torch.nn.Module):
    """crop to the same number of frames"""

    def __init__(self, clip_ratio=1 ):
        super().__init__()
        assert 0 < clip_ratio <= 1
        self.clip_ratio = clip_ratio

    def __call__(self, results):
        num_frames = int(results['total_frames'])
        clip_len = int(num_frames * self.clip_ratio)
        start = (num_frames - clip_len) // 2
        inds = np.arange(start, start + clip_len)
        results['total_frames'] = len(inds) # nuevo num de frames totales/ muestra
        keypoint = results['keypoint'][:, inds]
        name = [results['name'] [i] for i in inds]
        results['name'] = name
        results['keypoint'] = keypoint
        return results


class resize_frames(torch.nn.Module):
    """
    Resize the input video to the given clip length.
    Required keys are "keypoint", added or modified keys
    are "keypoint" and "frame_inds".
    Args:
        clip_len (int): clip length.
    """

    def __init__(self, clip_len=20):
        super().__init__()
        self.clip_len = clip_len

    def __call__(self, results):
        # frame_inds = results['frame_inds']
        # keypoint = results['keypoint'][:, frame_inds]
        keypoint = results['keypoint']
        m, t, v, c = keypoint.shape
        keypoint = keypoint.transpose((0, 3, 1, 2))  # M T V C -> M C T V
        keypoint = F.interpolate(
            torch.from_numpy(keypoint),
            size=(self.clip_len, v),
            mode='bilinear',
            align_corners=False)
        keypoint = keypoint.permute((0, 2, 3, 1)).numpy()
        # CTVM
        # keypoint = keypoint.permute((1, 2, 3, 0)).numpy()
        results['keypoint'] = keypoint
        results['total_frames'] = self.clip_len

        # inds = np.arange(self.clip_len)
        # results['frame_inds'] = inds.astype(np.int64)
        return results


"""create custom dataset by making our dataset a subclass of Pytorch Dataset"""


class CustomDataset(Dataset):
    def __init__(self, pickle_path, data_type='', centercrop=None, resizeframes=None):
        super().__init__()
        assert data_type in {'train', 'val', 'test'}, 'data type must be either train, val or test'
        self.pickle_path = pickle_path
        self.content = pd.read_pickle(self.pickle_path)
        self.type = data_type  # train ,val or test
        self.centercrop = centercrop
        self.resizeframes = resizeframes

    def __len__(self):
        """returns the total number of samples in the dataset"""
        # batches per epoch = samples/batch_size
        # content = pd.read_pickle(self.pickle_path)
        return len(self.content)
        # total_samples = len(content)
        # train_split = 0.85
        # train_len = int(total_samples * train_split)
        # val_len = total_samples - train_len
        # if self.type == 'train':
        #     # return train_len
        #     return len(content)
        # elif self.type == 'val':
        #     # return val_len
        #     return len(content)
        # elif self.type == 'test':
        #     return len(content)

    def __getitem__(self, idx):
        """ takes idx to select a sample from the dataset
        the method returns the sample ready to fed to the model"""
        # content = pd.read_pickle(self.pickle_path)
        # df = content[idx]['keypoint']  # 1,32,21,3
        # df = df.squeeze(axis=0)  # quitar la dim de batch
        # sample = torch.tensor(df)
        # sample = torch.permute(sample, (3, 1, 2, 0))
        label = self.content[idx]['label']
        sample = self.content[idx]
        sample = self.centercrop(sample)
        sample = self.resizeframes(sample)

        return sample
        # total_samples = len(content)  # 1529 total in train + val
        # train_split = 0.85
        # train_len = int(total_samples * train_split)
        # val_len = total_samples - train_len
        # train_set, val_set = torch.utils.data.random_split(content, [train_len, val_len])
        # train_set = content[:train_len]
        # val_set = content[train_len:]

        # if self.type == 'train':
        #     df = train_set[idx]['keypoint']  # 1,32,21,3
        #     # df = df.squeeze(axis=0)  # quitar la dim de batch
        #     sample = torch.tensor(df)
        #     sample = torch.permute(sample, (3, 1, 2, 0))
        #     label = train_set[idx]['label']
        #     if self.normalize:
        #         sample = self.normalize(sample)
        #     return sample, label
        #
        # if self.type == 'val':
        #     df = val_set[idx]['keypoint']  # 1,32,21,3
        #     # df = df.squeeze(axis=0)  # quitar la dim de batch
        #     sample = torch.tensor(df)
        #     sample = torch.permute(sample, (3, 1, 2, 0))
        #     label = val_set[idx]['label']
        #     if self.normalize:
        #         sample = self.normalize(sample)
        #     return sample, label

        #
        # if self.type == 'test':
        #     df = content[idx]['keypoint']  # 1,32,21,3
        #     # df = df.squeeze(axis=0)  # quitar la dim de batch
        #     sample = torch.tensor(df)
        #     sample = torch.permute(sample, (3, 1, 2, 0))
        #     label = content[idx]['label']
        #     if self.normalize:
        #         sample = self.normalize(sample)
        #     return sample, label

briareo_rgb_train = '/home/enz/code/skelact/data/briareo/xsub/train.pkl'
briareo_rgb_test = '/home/enz/code/skelact/data/briareo/xsub/test.pkl'
briareo_rgb_val = '/home/enz/code/skelact/data/briareo/xsub/val.pkl'

train_data = CustomDataset( pickle_path = briareo_rgb_train,
                           data_type='train',
                           centercrop=center_crop(clip_ratio=1),
                           resizeframes=resize_frames(clip_len= 10)
                           )

BATCH = 256
train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=0)
x_train = next(iter(train_dataloader))


