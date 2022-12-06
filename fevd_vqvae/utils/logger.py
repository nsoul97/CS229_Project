import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from fevd_vqvae.utils.dataset import unnormalize
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

    def _log_videos(self, video_frames, name, step, fps):
        assert len(video_frames.shape) == 5 and video_frames.shape[2] == 3, \
            "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video(name, video_frames, step, fps=fps)

    def _log_img_grid(self, images, name, step, nrow):
        assert len(images.shape) == 4 and images.shape[1] == 3, \
            "Need [N, C, H, W] input tensor for an image grid!"
        assert images.shape[0] == nrow * 2, "The input images must be stacked with their corresponding "
        img_grid = make_grid(images, nrow=nrow)
        self._summ_writer.add_image(name, img_grid, step)

    def _log_partition_visualizations(self, parition_name, input_videos, reconstruction_videos, input_imgs,
                                      reconstruction_imgs, step, fps, nrow):

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
            self._log_videos(video, f"{parition_name}/input_vs_reconstruction_video_{i}", step, fps)

        for i in range(0, len(input_imgs), nrow):
            input_grid_imgs = input_imgs[i:i+nrow]
            reconstruction_grid_imgs = reconstruction_imgs[i:i+nrow]
            grid = torch.cat((input_grid_imgs, reconstruction_grid_imgs), dim=0)
            self._log_img_grid(grid, f"{parition_name}/images/input_vs_reconstruction_img_grid_{i}", step, nrow=nrow)

    def log_loss(self, train_loss, val_loss, step):
        self._log_scalar(train_loss, "train/loss", step)
        self._log_scalar(val_loss, "val/loss", step)

        self._summ_writer.flush()

    def log_metrics(self,
                    val_fvd, val_lpips, val_psnr, val_ssim,
                    test_fvd, test_lpips, test_psnr, test_ssim,
                    step):

        self._log_scalar(val_fvd, "val/fvd", step)
        self._log_scalar(val_lpips, "val/lpips", step)
        self._log_scalar(val_psnr, "val/psnr", step)
        self._log_scalar(val_ssim, "val/ssim", step)

        self._log_scalar(test_fvd, "test/fvd", step)
        self._log_scalar(test_lpips, "test/lpips", step)
        self._log_scalar(test_psnr, "test/psnr", step)
        self._log_scalar(test_ssim, "test/ssim", step)

        self._summ_writer.flush()

    def log_visualizations(self,
                          train_input_videos, train_reconstruction_videos, train_input_imgs, train_reconstruction_imgs,
                          val_input_videos, val_reconstruction_videos, val_input_imgs, val_reconstruction_imgs,
                          test_input_videos, test_reconstruction_videos, test_input_imgs, test_reconstruction_imgs,
                          step, normalized_statistics=True, fps=10, nrow=4):

        self._log_partition_visualizations("train",
                                           train_input_videos, train_reconstruction_videos,
                                           train_input_imgs, train_reconstruction_imgs,
                                           step, normalized_statistics, fps, nrow)

        self._log_partition_visualizations("val",
                                           val_input_videos, val_reconstruction_videos,
                                           val_input_imgs, val_reconstruction_imgs,
                                           step, normalized_statistics, fps, nrow)

        self._log_partition_visualizations("test",
                                           test_input_videos, test_reconstruction_videos,
                                           test_input_imgs, test_reconstruction_imgs,
                                           step, normalized_statistics, fps, nrow)

        self._summ_writer.flush()


