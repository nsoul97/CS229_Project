import torch
import math


class PSNR:
    def __init__(self,
                 max_val: float = 1.) -> None:
        """
        The PSNR for a single frame is computed as 10 * log10( MAX ^ 2 / MSE), where:
        - MAX: the maximum value possible pixel value of an image (e.g. 1 or 255)
        - MSE: the mean squared error of the pixel values of the image

        To extend the PSNR for a video, the average PSNR is computed for all the frames of the video.

        :param max_val: The maximum possible pixel value of an image.
        """
        self._log10_max_val = 20 * math.log10(max_val)

    def __call__(self,
                 real_videos: torch.Tensor,
                 gen_videos: torch.Tensor) -> torch.Tensor:
        """
        Compute the Peak-Signal-to-Noise-Ratio metric for a set of real and generated videos.

        :param real_videos: A tensor with shape [B, T, C, H, W] containing T frames of B real videos
        :param gen_videos: A tensor with shape [B, T, C, H, W] containing T frames of the corresponding B generated videos.
        :return: A 1D tensor with length B, where the i-th value of the tensor indicates the PSNR of the i-th video.
        """

        mse = torch.mean((real_videos - gen_videos) ** 2, dim=(2, 3, 4))
        psnr_score_per_frame = self._log10_max_val - 10 * torch.log10(mse)
        psnr_score_per_video = torch.mean(psnr_score_per_frame, dim=1)
        return psnr_score_per_video


"""import tensorflow as tf
import numpy as np
import torch

# B , T, C, H, W
real_vid = np.random.rand(2, 4, 3, 16, 16)
gen_vid = np.random.rand(2, 4, 3, 16, 16)

real_vid_pt = torch.from_numpy(real_vid)
gen_vid_pt = torch.from_numpy(gen_vid)
mine_pnsr = PSNR(max_val=1)(real_vid_pt, gen_vid_pt)
mine_pnsr = [score.item() for score in mine_pnsr]

real_vid = real_vid.transpose([0, 1, 3, 4, 2]).reshape([-1, 16, 16, 3])
gen_vid = gen_vid.transpose([0, 1, 3, 4, 2]).reshape([-1, 16, 16, 3])

tf_psnr_per_frame = tf.image.psnr(real_vid, gen_vid, 1.0).numpy().reshape([2, 4])
tf_psnr = np.mean(tf_psnr_per_frame, axis=1)
print(mine_pnsr, tf_psnr)"""
