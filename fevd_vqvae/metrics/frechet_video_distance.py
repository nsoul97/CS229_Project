import scipy
from typing import Tuple, Union
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import numpy as np
import os


class InceptionI3d(nn.Module):

    def __init__(self):
        super(InceptionI3d, self).__init__()
        i3d_checkpoint = os.path.realpath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "checkpoints", "metrics",
                         "i3d_torchscript.pt"))
        self._i3d_kwargs = dict(rescale=False, resize=False, return_features=True)
        self._i3d_model = torch.jit.load(i3d_checkpoint)
        self._target_resolution = [224, 224]

    def forward(self, videos):
        B, T, C, H, W = videos.shape

        frames = videos.reshape(-1, C, H, W)
        videos = F.resize(frames, self._target_resolution).reshape(B, T, C, *self._target_resolution).transpose(1, 2)
        videos = 2. * videos - 1.
        feats = self._i3d_model(videos, **self._i3d_kwargs)

        return feats


class FVD:

    def __init__(self,
                 inception_i3d_model: nn.Module,
                 batch_size: int):
        self._inception_i3d_model = inception_i3d_model
        self._batch_size = batch_size

        self._idx = 0
        self._feats_fake = np.zeros((batch_size, 400))
        self._feats_real = np.zeros((batch_size, 400))

    def _reset(self):
        self._idx = 0
        self._feats_fake = np.zeros((self._batch_size, 400))
        self._feats_real = np.zeros((self._batch_size, 400))

    def _features_fvd(self,
                      feats_fake: np.ndarray,
                      feats_real: np.ndarray) -> float:
        mu_gen, sigma_gen = self._compute_stats(feats_fake)
        mu_real, sigma_real = self._compute_stats(feats_real)

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

    def _compute_stats(self,
                       feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0)  # [d]
        sigma = np.cov(feats, rowvar=False)  # [d, d]
        return mu, sigma

    def __call__(self,
                 videos_real: torch.Tensor,
                 videos_fake: torch.Tensor) -> Union[float, None]:
        """

        :param videos_fake: A tensor with shape [B, T, C, H, W] containing T frames of B fake videos
        :param videos_real: A tensor with shape [B, T, C, H, W] containing T frames of B real videos
        :return: The FVD for the given real and generated videos.
        """
        assert videos_real.shape[0] == videos_fake.shape[0], "The number of real and generated videos must be equal."
        B = videos_real.shape[0]

        feats_fake = self._inception_i3d_model(videos_fake).detach().cpu().numpy()
        feats_real = self._inception_i3d_model(videos_real).detach().cpu().numpy()

        self._feats_fake[self._idx: self._idx+B] = feats_fake
        self._feats_real[self._idx: self._idx+B] = feats_real

        self._idx += B
        if self._idx == self._batch_size:
            fvd = self._features_fvd(self._feats_fake, self._feats_real)
            self._reset()
            return fvd
        else:
            return None
