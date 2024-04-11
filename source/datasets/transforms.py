# Copyright (c) Hikvision Research Institute. All rights reserved.
import numpy
import numpy as np
import torch
import torch.nn.functional as F

from mmaction.datasets.builder import PIPELINES

@PIPELINES.register_module()
class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required Keys:

        - total_frames
        - start_index (optional)

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips
        - clip_len

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        seed (int): The random seed used during test time. Defaults to 255.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False,
                 seed: int = 255) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for training clips.
        """
        all_inds = []
        for clip_idx in range(self.num_clips):
            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

@PIPELINES.register_module()
class PoseRandomCrop:
    """Sample a clip randomly with a random clip length ranging from min_ratio
    to max_ratio from the video.

    Required keys are "total_frames", added or modified key is "frame_inds".

    Args:
        min_ratio (float): Minimal sampling ratio.
        max_ratio (float): Maximal sampling ratio.
        min_len (int): Minimal length of each sampled output clip.
    """

    def __init__(self, min_ratio=0.5, max_ratio=1.0, min_len=64):
        assert 0 < min_ratio <= 1
        assert 0 < max_ratio <= 1
        assert min_ratio <= max_ratio

        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_len = min_len

    def __call__(self, results):
        num_frames = results['total_frames']

        min_frames = int(num_frames * self.min_ratio)
        max_frames = int(num_frames * self.max_ratio)
        clip_len = np.random.randint(min_frames, max_frames + 1)
        clip_len = min(max(clip_len, self.min_len), num_frames)

        start = np.random.randint(0, num_frames - clip_len + 1)
        inds = np.arange(start, start + clip_len)

        results['frame_inds'] = inds.astype(np.int64)
        return results


@PIPELINES.register_module()
class PoseCenterCrop:
    """Sample a clip from the video.

    Required keys are "total_frames", added or modified key is "frame_inds".

    Args:
        clip_ratio (float): Sampling ratio.
    """

    def __init__(self, clip_ratio=0.9):
        assert 0 < clip_ratio <= 1
        self.clip_ratio = clip_ratio

    def __call__(self, results):
        num_frames = results['total_frames']

        clip_len = int(num_frames * self.clip_ratio)
        start = (num_frames - clip_len) // 2
        inds = np.arange(start, start + clip_len)

        results['frame_inds'] = inds.astype(np.int64)
        return results


@PIPELINES.register_module()
class PoseResize:
    """Resize the input video to the given clip length.
    Required keys are "keypoint", added or modified keys
    are "keypoint" and "frame_inds".
    Args:
        clip_len (int): clip length.
    """

    def __init__(self, clip_len=40):
        self.clip_len = clip_len

    def __call__(self, results):
        frame_inds = results['frame_inds']
        # keypoint = results['keypoint'][..., None]
        # results['keypoint'] = keypoint
        keypoint = results['keypoint'][frame_inds, :]
        keypoint = keypoint[None, ...]

        m, t, v, c = keypoint.shape
        keypoint = keypoint.transpose((0, 3, 1, 2))  # M T V C -> M C T V
        keypoint = F.interpolate(
            torch.from_numpy(keypoint),
            size=(self.clip_len, v),
            mode='bilinear',
            align_corners=False)
        # keypoint = keypoint.permute((0, 2, 3, 1)).numpy()
        # CTVM
        keypoint = keypoint.permute((1, 2, 3, 0)).numpy()
        results['keypoint'] = keypoint

        inds = np.arange(self.clip_len)
        results['frame_inds'] = inds.astype(np.int64)
        # files = []
        # for idx in inds:
        #     f = results['name'][idx]
        #     files.append(f)
        # print(len(files), files)
        return results


@PIPELINES.register_module()
class PoseRandomRotate:
    """Random rotate the input skeleton sequence.

    Required key is "keypoint", modified key is "keypoint".

    Args:
        rand_rotate (float): strength of rotation.
    """

    def __init__(self, rand_rotate=0.1):
        self.theta = rand_rotate * np.pi

    def __call__(self, results):
        keypoint = results['keypoint']

        keypoint = keypoint.transpose((3, 1, 2, 0))  # M T V C -> C T V M
        keypoint = self.random_rotate(keypoint)
        keypoint = keypoint.transpose((3, 1, 2, 0))  # C T V M -> M T V C
        results['keypoint'] = keypoint

        return results

    def random_rotate(self, keypoint):
        theta = np.random.uniform(-self.theta, self.theta, 3)
        cos = np.cos(theta)
        sin = np.sin(theta)

        # rotate by Z
        rot_z = np.eye(3)
        cos_z, sin_z = cos[0], sin[0]
        rot_z[0, 0] = cos_z
        rot_z[1, 0] = sin_z
        rot_z[0, 1] = -sin_z
        rot_z[1, 1] = cos_z

        # rotate by Y
        rot_y = np.eye(3)
        cos_y, sin_y = cos[1], sin[1]
        rot_y[0, 0] = cos_y
        rot_y[0, 2] = sin_y
        rot_y[2, 0] = -sin_y
        rot_y[2, 2] = cos_y

        # rotate by X
        rot_x = np.eye(3)
        cos_x, sin_x = cos[2], sin[2]
        rot_x[1, 1] = cos_x
        rot_x[2, 1] = sin_x
        rot_x[1, 2] = -sin_x
        rot_x[2, 2] = cos_x
        rot = np.matmul(np.matmul(rot_z, rot_y), rot_x)

        c, t, v, m = keypoint.shape
        keypoint = np.matmul(rot, keypoint.reshape(c, -1))
        keypoint = keypoint.reshape(c, t, v, m).astype(np.float32)
        return keypoint


@PIPELINES.register_module()
class IntToLong:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        labels = results[self.keys[1]]
        labels = labels.astype(dtype=numpy.int64)
        # labels = labels.type(torch.LongTensor)
        results[self.keys[1]] = labels

        return results
