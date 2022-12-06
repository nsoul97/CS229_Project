import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from fevd_vqvae.utils.dataset import unnormalize
from typing import Dict
import os
import shutil


class Logger:
    def __init__(self,
                 log_dir_path: str,
                 log_file_name: str,
                 resume: bool) -> None:

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)

        log_file_path = os.path.join(log_dir_path, log_file_name)

        path_exists = os.path.exists(log_file_path)
        if resume:
            assert path_exists, "Cannot resume training. Previous logs do not exist."
        elif path_exists:
            shutil.rmtree(log_file_path)

        self._summ_writer = SummaryWriter(log_file_path)

    def _log_scalar(self, scalar, name, step):
        self._summ_writer.add_scalar(name, scalar, step)

    def _log_videos(self, video_frames, name, step):
        assert len(video_frames.shape) == 5 and video_frames.shape[2] == 3, \
            "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video(name, video_frames, step, fps=10)

    def _log_img_grid(self, images, name, step, nrow):
        assert len(images.shape) == 4 and images.shape[1] == 3, \
            "Need [N, C, H, W] input tensor for an image grid!"
        assert images.shape[0] == nrow * 2, "The input images must be stacked with their corresponding "
        img_grid = make_grid(images, nrow=nrow)
        self._summ_writer.add_image(name, img_grid, step)

    def log_visualizations(self, split,
                           input_videos, reconstruction_videos,
                           input_imgs, reconstruction_imgs,
                           step, nrow):

        assert input_videos.shape == reconstruction_videos.shape, "The input videos and the reconstruction videos must " \
                                                                  "have the same shape."

        assert input_imgs.shape == reconstruction_imgs.shape, "The input images and the reconstruction images must " \
                                                              "have the same shape."

        assert input_imgs.shape[0] % nrow == 0, f"{input_imgs.shape[0]} images were given, but each row in the image " \
                                                f"grid contains {nrow} images."

        input_videos = unnormalize(input_videos)
        torch.clamp(input_videos, 0., 1.)

        reconstruction_videos = unnormalize(reconstruction_videos)
        torch.clamp(reconstruction_videos, 0., 1.)

        input_imgs = unnormalize(input_imgs)
        torch.clamp(input_imgs, 0., 1.)

        reconstruction_imgs = unnormalize(reconstruction_imgs)
        torch.clamp(reconstruction_imgs, 0., 1.)

        for i in range(len(input_videos)):
            input_video = input_videos[i]
            reconstruction_video = reconstruction_videos[i]
            video = torch.stack((input_video, reconstruction_video), dim=0)
            self._log_videos(video, f"{split}/input_vs_reconstruction_video_{i}", step)

        for i in range(0, len(input_imgs), nrow):
            input_grid_imgs = input_imgs[i:i + nrow]
            reconstruction_grid_imgs = reconstruction_imgs[i:i + nrow]
            grid = torch.cat((input_grid_imgs, reconstruction_grid_imgs), dim=0)
            self._log_img_grid(grid, f"{split}/images/input_vs_reconstruction_img_grid_{i}", step, nrow=nrow)

        self._summ_writer.flush()

    def log_dict(self,
                 log_dict: Dict[str, float],
                 split: str,
                 step: int):
        """ Logs complete the loss/metrics dict for a given step
        
        Args:
            log_dict (dict): Dict containing training image losses or the evaluation metrics
            split(str): 'train', 'val' or 'train'
            step (int): Current training step
        """
        for k, v in log_dict.items():
            self._log_scalar(v, f"{split}/{k}", step)

        self._summ_writer.flush()
