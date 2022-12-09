import torch.utils.data as data
import torch
import numpy as np
import pandas as pd
import omegaconf
from typing import List, Union, Dict, Optional
from torch.types import Device
import os

_statistics = {'mean': [0.4947905177456989, 0.4394289448514622, 0.3734249455050442],
               'std': [0.2248808632122409, 0.2230480187693433, 0.2166628109431439]}


def _mean_std_2_tensors(mean: List[float],
                        std: List[float],
                        target_shape_dim: int,
                        device: Device):
    mean = torch.tensor(mean, device=device)
    std = torch.tensor(std, device=device)

    if target_shape_dim == 3:  # image single example (C, H, W)
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)

    elif target_shape_dim == 4:  # images batched examples (B, C, H, W) | video single example: (T, C, H, W)
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)

    elif target_shape_dim == 5:  # videos batched examples (B, T, C, H, W)
        mean = mean.reshape(1, 1, -1, 1, 1)
        std = std.reshape(1, 1, -1, 1, 1)

    return mean, std


def unnormalize(frames: torch.Tensor):
    mean, std = _mean_std_2_tensors(_statistics['mean'], _statistics['std'], len(frames.shape), frames.device)
    frames = frames * std + mean
    return frames


def normalize(frames: torch.Tensor):
    mean, std = _mean_std_2_tensors(_statistics['mean'], _statistics['std'], len(frames.shape), frames.device)
    frames = (frames - mean) / std
    return frames


def _setup_split_dataloader(root_dir_path: str,
                            dataset_module: str,
                            split: str,
                            num_workers: int,
                            batch_size: int,
                            shuffle: bool,
                            drop_last: bool,
                            use_only_videos: Union[int, None] = None):

    if dataset_module == 'RobonetImgDataset':
        dataset = RobonetImgDataset(root_dir_path=root_dir_path, split=split, use_only_videos=use_only_videos)
    else:
        dataset = RobonetVideoDataset(root_dir_path=root_dir_path, split=split, use_only_videos=use_only_videos)

    return data.DataLoader(dataset,
                           num_workers=num_workers,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           pin_memory=True)


def setup_dataloader(root_dir_path: str,
                     dataset_module: str,
                     split: Union[List[str], str],
                     num_workers: int,
                     batch_size: int,
                     shuffle: bool,
                     drop_last: bool,
                     use_only_videos: Optional[int] = None) -> Union[data.DataLoader, Dict[str, data.DataLoader]]:

    if (not isinstance(split, str)) and (not isinstance(split, omegaconf.listconfig.ListConfig)):
        raise TypeError(f"The split can be either str or list. not {type(split)}")

    if dataset_module != 'RobonetImgDataset' and dataset_module != 'RobonetVideoDataset':
        raise NameError(f"The dataset_module can be either 'RobonetImgDataset' or 'RobonetVideoDataset', not "
                        f"{dataset_module}")

    if isinstance(split, str):
        return _setup_split_dataloader(root_dir_path, dataset_module, split, num_workers, batch_size, shuffle,
                                       drop_last, use_only_videos)
    else:
        return {s: _setup_split_dataloader(root_dir_path, dataset_module, s, num_workers, batch_size, shuffle,
                                           drop_last, use_only_videos) for s in split}


class RobonetImgDataset(data.Dataset):

    def __init__(self,
                 root_dir_path: str,
                 split: str,
                 use_only_frames: int = 12,
                 use_only_videos: Union[int, None] = None):

        super(RobonetImgDataset, self).__init__()

        assert split in ['train', 'val', 'test'], "The split can be either 'train', 'val' or 'test'."

        self._split_dir_path = os.path.join(root_dir_path, "video_data", split)
        video_files = os.listdir(self._split_dir_path)
        if use_only_videos is not None: video_files = video_files[:use_only_videos]

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

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_name, frame = self._data[idx]
        file_path = os.path.join(self._split_dir_path, file_name)
        video = np.load(file_path).astype(float)
        frame = torch.from_numpy(video[frame])
        frame = torch.permute(frame, [2, 0, 1]).to(memory_format=torch.contiguous_format).float()
        frame /= 255.
        frame = normalize(frame)
        return frame


class RobonetVideoDataset(data.Dataset):
    def __init__(self,
                 root_dir_path: str,
                 split: str,
                 use_only_frames: int = 12,
                 total_frames: int = 12,
                 use_only_videos: Union[int, None] = None):

        super(RobonetVideoDataset, self).__init__()

        assert split in ['train', 'val', 'test'], "The split can be either 'train', 'val' or 'test'."
        assert (use_only_frames is None) or (use_only_frames >= total_frames), f"The total frames that are used for " \
                                                                               f"video prediction cannot be less " \
                                                                               f"than the frames that we are using."

        self._split_dir_path = os.path.join(root_dir_path, "video_data", split)
        video_files = os.listdir(self._split_dir_path)
        if use_only_videos is not None: video_files = video_files[:use_only_videos]

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

        self._total_frames = total_frames

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        file_name, start_frame_idx = self._data[idx]
        file_path = os.path.join(self._split_dir_path, file_name)
        video = np.load(file_path)
        video_segment = torch.from_numpy(video[start_frame_idx: start_frame_idx + self._total_frames])
        video_segment = torch.permute(video_segment, [0, 3, 1, 2]).to(memory_format=torch.contiguous_format).float()
        video_segment /= 255.
        video_segment = normalize(video_segment)
        return video_segment
