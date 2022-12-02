import torch.utils.data as data
import torch
import torchvision.transforms.functional as F
import numpy as np
import pandas as pd
import os

_statistics = {'mean': [0.49479052, 0.43942894, 0.37342495],
               'std': [0.22488086, 0.22304802, 0.21666281]}


def unnormalize(frames: torch.Tensor,
                normalize_statistics: bool = True):

    if normalize_statistics:
        frames.mul_(_statistics['std']).add_(_statistics['mean'])
    frames.clamp_(min=0., max=1.)
    return frames


class RobonetImgDataset(data.Dataset):

    def __init__(self,
                 root_dir_path: str,
                 split: str,
                 use_only_frames: int = None,
                 normalize_statistics: bool = True):

        assert split in ['train', 'val', 'test'], "The split can be either 'train', 'val' or 'test'."

        self._split_dir_path = os.path.join(root_dir_path, "video_data", split)
        video_files = os.listdir(self._split_dir_path)

        metadata_file_path = os.path.join(root_dir_path, "metadata.csv")
        metadata = pd.read_csv(metadata_file_path, index_col=0, low_memory=False)
        traj_frames = metadata['state_T']

        self._data = []
        for video_fname in video_files:
            traj_name = '_'.join(video_fname.split('_')[:-1]) + '.hdf5'
            video_frames = traj_frames.loc[traj_name]
            if use_only_frames is None:
                self._data += list(zip([video_fname] * video_frames, list(range(video_frames))))
            else:
                self._data += list(zip([video_fname] * use_only_frames, list(range(use_only_frames))))

        self._normalize_statistics = normalize_statistics

    def __len__(self):
        return len(self._data)

    def _normalize(self,
                   frame: torch.Tensor):
        frame /= 255.
        if self._normalize_statistics:
            frame = F.normalize(frame, mean=_statistics['mean'], std=_statistics['std'])
        return frame

    def __getitem__(self, idx):
        file_name, frame = self._data[idx]
        file_path = os.path.join(self._split_dir_path, file_name)
        video = np.load(file_path).astype(float)
        frame = torch.from_numpy(video[frame])
        frame = torch.permute(frame, [2, 0, 1])
        frame = self._normalize(frame)
        return frame


class RobonetVideoDataset(data.Dataset):
    def __init__(self,
                 root_dir_path: str,
                 split: str,
                 use_only_frames: int = None,
                 normalize_statistics: bool = True,
                 total_frames: int = 12):

        assert split in ['train', 'val', 'test'], "The split can be either 'train', 'val' or 'test'."
        assert (use_only_frames is None) or (use_only_frames >= total_frames), f"The total frames that are used for " \
                                                                               f"video prediction cannot be less "    \
                                                                               f"than the frames that we are using."

        self._split_dir_path = os.path.join(root_dir_path, "video_data", split)
        video_files = os.listdir(self._split_dir_path)

        metadata_file_path = os.path.join(root_dir_path, "metadata.csv")
        metadata = pd.read_csv(metadata_file_path, index_col=0, low_memory=False)
        traj_frames = metadata['state_T']

        self._data = []
        for video_fname in video_files:
            traj_name = '_'.join(video_fname.split('_')[:-1]) + '.hdf5'
            video_frames = traj_frames.loc[traj_name]
            if use_only_frames is None:
                video_num_windows = video_frames - total_frames + 1
            else:
                video_num_windows = use_only_frames - total_frames + 1

            self._data += list(zip([video_fname] * video_num_windows, list(range(video_num_windows))))

        self._normalize_statistics = normalize_statistics
        self._total_frames = total_frames

    def __len__(self):
        return len(self._data)

    def _normalize(self,
                   video_segment: torch.Tensor):
        video_segment /= 255.
        if self._normalize_statistics:
            video_segment = F.normalize(video_segment, mean=_statistics['mean'], std=_statistics['std'])
        return video_segment

    def __getitem__(self, idx):
        file_name, start_frame_idx = self._data[idx]
        file_path = os.path.join(self._split_dir_path, file_name)
        video = np.load(file_path).astype(float)
        video_segment = torch.from_numpy(video[start_frame_idx: start_frame_idx+self._total_frames])
        video_segment = self._normalize(video_segment)
        return video_segment


root_dir_path = "/home/soul/Development/Stanford/Fall 2022/CS 229: Machine Learning/Project/data/dataset/preprocessed_data"
dataset = RobonetImgDataset(root_dir_path=root_dir_path,
                        split="train")

from tqdm import tqdm
true_mean = _statistics['mean'].reshape(-1, 1, 1)
rgb_mean = torch.zeros(3)
rgb_variance = torch.zeros(3)
for i in tqdm(range(len(dataset))):
    img = dataset[i]
    rgb_mean += torch.mean(img, dim=[1, 2]) / len(dataset)
    rgb_variance += torch.mean((img - true_mean)**2, dim=[1, 2]) / len(dataset)
rgb_std = torch.sqrt(rgb_variance)

print(rgb_mean, rgb_std)

