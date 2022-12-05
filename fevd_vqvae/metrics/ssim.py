import torch
import torch.nn.functional as F
import torch.types


class SSIM:
    def __init__(self,
                 device: torch.types.Device,
                 max_val: float = 1.,
                 filter_size: int = 11,
                 filter_sigma: float = 1.5,
                 k1: float = 0.01,
                 k2: float = 0.03) -> None:
        self._max_val = max_val
        self._filter_size = filter_size
        self._filter_sigma = filter_sigma
        self._k1 = k1
        self._k2 = k2

        self._calc_gaussian_kernel(device)

        self._c1 = (self._k1 * max_val) ** 2
        self._c2 = (self._k2 * max_val) ** 2

    def _calc_gaussian_kernel(self, device):
        coords = torch.arange(self._filter_size, device=device) - ((self._filter_size - 1) / 2.)
        g = - coords ** 2 / (2 * self._filter_sigma ** 2)
        g = g.view(1, -1) + g.view(-1, 1)
        g = torch.exp(g)
        self._gaussian_kernel = g / torch.sum(g)
        self._gaussian_kernel = self._gaussian_kernel.view(1, 1, 1, self._filter_size, self._filter_size, 1).repeat(1, 1, 3, 1, 1, 1)

    def __call__(self,
                 real_videos: torch.Tensor,
                 gen_videos: torch.Tensor) -> torch.Tensor:
        """
        :param real_videos: A tensor with shape [B, T, C, H, W] containing T frames of B real videos
        :param gen_videos: A tensor with shape [B, T, C, H, W] containing T frames of the corresponding B generated videos.
        :return: A 1D tensor with length B, where the i-th value of the tensor indicates the PSNR of the i-th video.
        """

        B, T, C, H, W = real_videos.shape

        real_videos = real_videos.view(-1, C, H, W)
        gen_videos = gen_videos.view(-1, C, H, W)

        real_videos_patches = F.unfold(real_videos, self._filter_size).reshape(B, T, C, self._filter_size,
                                                                               self._filter_size, -1)
        gen_videos_patches = F.unfold(gen_videos, self._filter_size).reshape(B, T, C, self._filter_size,
                                                                             self._filter_size, -1)

        mx = torch.sum(real_videos_patches * self._gaussian_kernel, dim=(3, 4))
        mxx = torch.sum(real_videos_patches ** 2 * self._gaussian_kernel, dim=(3, 4))
        square_mx = mx ** 2
        cxx = mxx - square_mx

        my = torch.sum(gen_videos_patches * self._gaussian_kernel, dim=(3, 4))
        myy = torch.sum(gen_videos_patches ** 2 * self._gaussian_kernel, dim=(3, 4))
        square_my = my ** 2
        cyy = myy - square_my

        mxy = torch.sum(real_videos_patches * gen_videos_patches * self._gaussian_kernel, dim=(3, 4))
        prod_mx_my = mx * my
        cxy = mxy - prod_mx_my

        luminance = (2 * prod_mx_my + self._c1) / (square_mx + square_my + self._c1)
        contrast_structure = (2 * cxy + self._c2) / (cxx + cyy + self._c2)

        ssim_per_video = torch.mean(luminance * contrast_structure, dim=(1, 2, 3))
        return ssim_per_video


"""import tensorflow as tf
import numpy as np
import torch

# B , T, C, H, W
real_vid = np.random.rand(2, 4, 3, 16, 16)
gen_vid = np.random.rand(2, 4, 3, 16, 16)

real_vid_pt = torch.from_numpy(real_vid)
gen_vid_pt = torch.from_numpy(gen_vid)
mine_ssim = SSIM(max_val=1.0)(real_vid_pt, real_vid_pt)
mine_ssim = [score.item() for score in mine_ssim]

real_vid = real_vid.transpose([0, 1, 3, 4, 2]).reshape([-1, 16, 16, 3])
gen_vid = gen_vid.transpose([0, 1, 3, 4, 2]).reshape([-1, 16, 16, 3])

tf_ssim_per_frame = tf.image.ssim(real_vid, real_vid, 1.0).numpy().reshape([2, 4])
tf_ssim = np.mean(tf_ssim_per_frame, axis=1)
print(mine_ssim, tf_ssim)"""