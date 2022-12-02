import torch
import torch as th
import psnr
import ssim
import lpips_video
import frechet_video_distance as fvd

class MetricsTracker:
    def __init__(self,
                 max_val: int = 1,
                 normalize: bool = True) -> None:

        self.PSNR_metric = psnr.PSNR(max_val)
        self.SSIM_metric = ssim.SSIM(max_val)
        self.LPIPS_metric = lpips_video.LPIPS_Video(max_val, normalize)
        self.FVD_metric = fvd.FVD(max_val, normalize)
        self.reset()

    def reset(self) -> None:
        self._total_videos = 0
        self._total_updates = 0

        self._batch_psnr = 0
        self._batch_ssim = 0
        self._batch_lpips = 0
        self._batch_fvd = 0

    @torch.inference_mode()
    def update(self,
               real_videos: th.Tensor,
               gen_videos: th.Tensor) -> None:

        assert real_videos.shape == gen_videos.shape
        B = real_videos.shape[0]
        self._total_videos += B
        self._total_updates += 1

        mini_batch_video_psnr = self.PSNR_metric(real_videos, gen_videos)
        self._batch_psnr += th.sum(mini_batch_video_psnr)

        mini_batch_video_ssim = self.SSIM_metric(real_videos, gen_videos)
        self._batch_ssim += th.sum(mini_batch_video_ssim)

        mini_batch_video_lpips = self.LPIPS_metric(real_videos, gen_videos)
        self._batch_lpips += th.sum(mini_batch_video_lpips)

        mini_batch_fvd = self.FVD_metric(real_videos, gen_videos)
        self._batch_fvd += mini_batch_fvd

    def compute(self) -> dict:
        self._batch_psnr /= self._total_videos
        self._batch_ssim /= self._total_videos
        self._batch_lpips /= self._total_videos
        self._batch_fvd /= self._total_updates

        return {'PSNR': self._batch_psnr.item(),
                "SSIM": self._batch_ssim.item(),
                "LPIPS": self._batch_lpips.item(),
                "FVD": self._batch_fvd}


tracker = MetricsTracker()
for epoch in range(5):
    tracker.reset()
    for mb in range(20):
        real_videos = th.rand([64, 10, 3, 64, 64])
        gen_videos = real_videos * 0.75
        tracker.update(real_videos, gen_videos)

    epoch_metrics = tracker.compute()
    print(epoch_metrics)
