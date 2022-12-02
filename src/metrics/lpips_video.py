import lpips as lpips_frame
import torch as th

class LPIPS_Video:
    def __init__(self,
                 max_val: int,
                 normalize: bool = False,
                 device: str = 'cuda' if th.cuda.is_available() else 'cpu') -> None:
        """
        The LPIPS for a single frame is computed as the similarity between the activations of two real and the generated
        frame patches from the 5th layer of the VGG-network.

       To extend the LPIPS for a video, the average LPIPS is computed for all the frames of the video.

        :param max_val: The maximum possible pixel value of an image.
        :param normalize: If the range of the pixel values is not [-1, 1], normalize should be True.
        """
        self._max_val = max_val
        self._normalize = normalize
        self._lpips_per_frame_metric = lpips_frame.LPIPS().to(device)
        self._device = device

    def __call__(self,
                 real_videos: th.Tensor,
                 gen_videos: th.Tensor) -> th.Tensor:
        """
        Compute the Peak-Signal-to-Noise-Ratio metric for a set of real and generated videos.

        :param real_videos: A tensor with shape [B, T, C, H, W] containing T frames of B real videos
        :param gen_videos: A tensor with shape [B, T, C, H, W] containing T frames of the corresponding B generated videos.
        :return: A 1D tensor with length B, where the i-th value of the tensor indicates the PSNR of the i-th video.
        """

        B, T, C, H, W = real_videos.shape

        real_videos = real_videos.to(self._device)
        gen_videos = gen_videos.to(self._device)

        real_videos = real_videos.reshape(-1, C, H, W) / self._max_val
        gen_videos = gen_videos.reshape(-1, C, H, W) / self._max_val

        if self._normalize:
            real_videos = 2. * real_videos - 1.
            gen_videos = 2. * gen_videos - 1.

        real_videos = real_videos.float()
        gen_videos = gen_videos.float()

        lpips_per_frame = self._lpips_per_frame_metric(real_videos, gen_videos).reshape(B, T)
        lpips_per_video = th.mean(lpips_per_frame, dim=1)
        return lpips_per_video


"""import torch
import numpy as np

# B , T, C, H, W
real_vid = np.random.rand(2, 4, 3, 64, 64)
gen_vid = np.random.rand(2, 4, 3, 64, 64)

real_vid_pt = torch.from_numpy(real_vid)
gen_vid_pt = torch.from_numpy(gen_vid)

mine_lpips = LPIPS_Video(max_val=1.0)(real_vid_pt, gen_vid_pt)
mine_lpips = [score.item() for score in mine_lpips]

print(mine_lpips)

mine_lpips = LPIPS_Video(max_val=1.0)(real_vid_pt, gen_vid_pt)
mine_lpips = [score.item() for score in mine_lpips]

print(mine_lpips)"""