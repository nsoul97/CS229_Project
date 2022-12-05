import torch
import torch.nn as nn
from fevd_vqvae.utils.dataset import unnormalize
import fevd_vqvae.metrics.lpips as lpips
import fevd_vqvae.metrics.psnr as psnr
import fevd_vqvae.metrics.ssim as ssim
import fevd_vqvae.metrics.frechet_video_distance as fvd
import torch.types

class MetricsTracker:
    def __init__(self,
                 lpips_model: nn.Module,
                 inception_i3d_model: nn.Module,
                 batch_size: int,
                 device: torch.types.Device) -> None:

        self.PSNR_metric = psnr.PSNR()
        self.SSIM_metric = ssim.SSIM(device=device)
        self.LPIPS_metric = lpips.LPIPS_Video(lpips_model=lpips_model)
        self.FVD_metric = fvd.FVD(inception_i3d_model=inception_i3d_model,
                                  batch_size=batch_size)
        self._batch_size = batch_size
        self._reset()

    def _reset(self) -> None:
        self._total_videos = 0
        self._batch_psnr = 0
        self._batch_ssim = 0
        self._batch_lpips = 0

    def update(self,
               real_videos: torch.Tensor,
               gen_videos: torch.Tensor) -> None:

        assert real_videos.shape == gen_videos.shape
        B = real_videos.shape[0]
        self._total_videos += B

        real_videos = unnormalize(real_videos)
        torch.clamp(real_videos, 0., 1.)

        gen_videos = unnormalize(gen_videos)
        torch.clamp(gen_videos, 0., 1.)

        mini_batch_video_psnr = self.PSNR_metric(real_videos, gen_videos)
        self._batch_psnr += torch.sum(mini_batch_video_psnr)

        mini_batch_video_ssim = self.SSIM_metric(real_videos, gen_videos)
        self._batch_ssim += torch.sum(mini_batch_video_ssim)

        mini_batch_video_lpips = self.LPIPS_metric(real_videos, gen_videos)
        self._batch_lpips += torch.sum(mini_batch_video_lpips)

        self._batch_fvd = self.FVD_metric(real_videos, gen_videos)

    def compute(self) -> dict:

        assert self._total_videos == self._batch_size, f"The statistics can only be computed after a full mini-batch of " \
                                                       f"size {self._batch_size} (currently total_videos={self._total_videos})."

        self._batch_psnr /= self._total_videos
        self._batch_ssim /= self._total_videos
        self._batch_lpips /= self._total_videos

        log =  {'PSNR': self._batch_psnr.item(),
                "SSIM": self._batch_ssim.item(),
                "LPIPS": self._batch_lpips.item(),
                "FVD": self._batch_fvd}

        self._reset()
        return log


"""tracker = MetricsTracker()
for epoch in range(5):
    tracker.reset()
    for mb in range(20):
        real_videos = th.rand([64, 10, 3, 64, 64])
        gen_videos = real_videos * 0.75
        tracker.update(real_videos, gen_videos)

    epoch_metrics = tracker.compute()
    print(epoch_metrics)"""
