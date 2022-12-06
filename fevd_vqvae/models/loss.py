import torch
import torch.nn as nn
from fevd_vqvae.metrics.lpips import LPIPS
from fevd_vqvae.metrics.frechet_video_distance import InceptionI3d
from fevd_vqvae.utils.dataset import unnormalize


class VQLoss(nn.Module):
    def __init__(self,
                 codebook_weight=1.0,
                 pixelloss_weight=1.0,
                 perceptual_weight_2d=1.0,
                 fvd_mu_weight=1.0,
                 fvd_cov_weight=1.0):
        super().__init__()

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight_2d = perceptual_weight_2d
        self.fvd_mu_weight = fvd_mu_weight
        self.fvd_cov_weight = fvd_cov_weight

        self.perceptual_loss_2d = LPIPS().eval()
        self.inception_i3d = InceptionI3d().eval()

    def _get_fvd_loss(self, real_videos, rec_videos):

        assert len(real_videos.shape) == 5, f"The FVD metric can be computed only for a batch of videos. " \
                                            f"(input shape: {real_videos.shape}"

        real_videos = unnormalize(real_videos)
        rec_videos = unnormalize(rec_videos)

        real_vid_feats = self.inception_i3d(real_videos)
        rec_vid_feats = self.inception_i3d(rec_videos)

        mu_real = torch.mean(real_vid_feats, dim=0)
        mu_rec = torch.mean(rec_vid_feats, dim=0)
        fvd_loss_mu = torch.mean((mu_real - mu_rec)**2)

        cov_real = torch.cov(real_vid_feats.T)
        cov_rec = torch.cov(rec_vid_feats.T)
        fvd_loss_cov = torch.mean((cov_real - cov_rec)**2)

        fvd_loss_mu = fvd_loss_mu.reshape(1, 1, 1, 1, 1)
        fvd_loss_cov = fvd_loss_cov.reshape(1, 1, 1, 1, 1)

        return fvd_loss_mu, fvd_loss_cov

    def forward(self, codebook_loss, inputs, reconstructions):

        rec_loss = torch.zeros_like(inputs, device=inputs.device)

        if self.pixel_weight > 0:
            pixel_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            rec_loss += self.pixel_weight * pixel_loss
        else:
            pixel_loss = torch.tensor([0.0])

        if self.perceptual_weight_2d > 0:
            perc_2d_loss = self.perceptual_loss_2d(inputs.contiguous(), reconstructions.contiguous())
            rec_loss += self.perceptual_weight_2d * perc_2d_loss
        else:
            perc_2d_loss = torch.tensor([0.0])

        fvd_loss = torch.tensor([0.0])
        fvd_loss_mu = torch.tensor([0.0])
        fvd_loss_cov = torch.tensor([0.0])
        if self.fvd_mu_weight > 0 or self.fvd_cov_weight > 0:
            fvd_loss_mu, fvd_loss_cov = self._get_fvd_loss(inputs, reconstructions)
            if self.fvd_mu_weight == 0: fvd_loss_mu = torch.tensor([0.0])
            if self.fvd_cov_weight == 0: fvd_loss_cov = torch.tensor([0.0])
            fvd_loss = self.fvd_mu_weight * fvd_loss_mu + self.fvd_cov_weight * fvd_loss_cov
            rec_loss += fvd_loss

        if self.codebook_weight > 0:
            loss = rec_loss.mean() + self.codebook_weight * codebook_loss.mean()
        else:
            loss = rec_loss.mean()
            codebook_loss = torch.tensor([0.0])

        log = {"total_loss": loss.mean().item(),
               "quant_loss": codebook_loss.mean().item(),
               "rec_loss": rec_loss.mean().item(),
               "pixel_loss": pixel_loss.mean().item(),
               "perc_2d_loss": perc_2d_loss.mean().item(),
               "fvd_loss": fvd_loss.mean().item(),
               "fvd_mu_loss": fvd_loss_mu.mean().item(),
               "fvd_cov_loss": fvd_loss_cov.mean().item()}

        return loss, log
