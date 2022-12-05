import torch
import torch.nn as nn
from fevd_vqvae.metrics.lpips import LPIPS
from fevd_vqvae.metrics.frechet_video_distance import InceptionI3d


class VQLoss(nn.Module):
    def __init__(self,
                 codebook_weight=1.0,
                 pixelloss_weight=1.0,
                 perceptual_weight_2d=1.0,
                 perceptual_weight_3d=1.0):
        super().__init__()

        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight_2d = perceptual_weight_2d
        self.perceptual_weight_3d = perceptual_weight_3d

        self.perceptual_loss_2d = LPIPS().eval()
        self.perceptual_loss_3d = InceptionI3d().eval()

    def forward(self, codebook_loss, inputs, reconstructions):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight_2d > 0:
            perc_2d_loss = self.perceptual_loss_2d(inputs.contiguous(), reconstructions.contiguous())
            rec_loss += self.perceptual_weight_2d * perc_2d_loss
        else:
            perc_2d_loss = torch.tensor([0.0])

        if self.perceptual_weight_3d > 0:
            perc_3d_loss = self.perceptual_loss_3d(inputs.contiguous(), reconstructions.contiguous())
            rec_loss += self.perceptual_weight_3d * perc_3d_loss
        else:
            perc_3d_loss = torch.tensor([0.0])

        nll_loss = torch.mean(rec_loss)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {"total_loss": loss.clone().detach().mean(),
               "quant_loss": codebook_loss.detach().mean(),
               "nll_loss": nll_loss.detach().mean(),
               "rec_loss": rec_loss.detach().mean(),
               "perc_2d_loss": perc_2d_loss.detach().mean(),
               "perc_3d_loss": perc_3d_loss.detach().mean()}

        return loss, log
