import torch
import torch.nn as nn
from fevd_vqvae.models.model_modules import Encoder, Decoder2D
from fevd_vqvae.models.vector_quantizer import VectorQuantizer
from fevd_vqvae.models.utils import instantiate_from_config
from fevd_vqvae.models.loss import VQLoss


class VQModel(nn.Module):
    def __init__(self,
                 key,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_sd=None):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.loss = VQLoss(**lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)

        if key == '2d':
            self.decoder = Decoder2D(**ddconfig)
            self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
            self.decode = self.decode_2d
        elif key == '3d':
            self.decoder = Decoder3D(**ddconfig)
            self.post_quant_conv = nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
            self.decode = self.decode_3d
            pass
        else:
            raise ValueError(f"The key should be either '2d' or '3d', not {key}")

        if ckpt_sd is not None:
            self.load_state_dict(ckpt_sd)

    def encode(self, x):

        input_is_videos = len(x.shape) == 5
        if input_is_videos:
            B, T, C, H, W = x.shape  # the input is a batch of videos (B, T, C, H, W)
            x = x.reshape(B * T, C, H, W)

        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)

        if input_is_videos:
            _, C, H, W = quant.shape
            quant = quant.reshape(B, T, C, H, W)
        return quant, emb_loss, info

    def decode_2d(self, quant):

        input_is_videos = len(quant.shape) == 5
        if input_is_videos:
            B, T, C, H, W = quant.shape  # the input is a batch of videos (B, T, C, H, W)
            quant = quant.reshape(B * T, C, H, W)

        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)

        if input_is_videos:
            _, C, H, W = dec.shape
            dec = dec.reshape(B, T, C, H, W)

        return dec

    def decode_3d(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


    def forward(self, real_videos):
        quant, codebook_loss, _ = self.encode(real_videos)
        rec_videos = self.decode(quant)
        return rec_videos, codebook_loss

    def step(self, real_videos):
        rec_videos, qloss = self(real_videos)
        loss, loss_dict = self.loss(qloss, real_videos, rec_videos)
        return rec_videos, loss, loss_dict

    def configure_optimizer(self, lr ,opt_sd=None):
        opt = torch.optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()) +
                               list(self.quantize.parameters()) +
                               list(self.quant_conv.parameters()) +
                               list(self.post_quant_conv.parameters()),
                               lr=lr,
                               betas=(0.5, 0.9))
        if opt_sd is not None:
            opt.load_state_dict(opt_sd)
        return opt
