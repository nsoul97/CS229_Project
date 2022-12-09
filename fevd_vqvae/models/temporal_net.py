import torch
import torch.nn as nn
try:
  from fevd_vqvae.models.pytorch_i3d import InceptionI3d
except ModuleNotFoundError:
  from models.pytorch_i3d import InceptionI3d
from torchvision import transforms

def transpose_channels(video):
  if len(video.shape) == 5:   # b, t, c, h, w  ->  b, c, t, h, w
    video = video.transpose(1, 2)
  else:                   # t, c, h, w -> c, t, h, w
    video = video.transpose(0, 1)
  return video

class TemporalModel(nn.Module):
  def __init__(self, i3d_ckpt= "rgb_imagenet.pt"):
    super(TemporalModel, self).__init__()
    self.resize = transforms.Resize((224, 224))
    self.i3d = InceptionI3d(400, in_channels=3)
    self.i3d.load_state_dict(torch.load(i3d_ckpt))
    for param in self.i3d.parameters():
      param.requires_grad = False
    self.i3d.logits = torch.nn.Identity()
    self.i3d.Mixed_4d = torch.nn.Identity()
    self.i3d.Mixed_4e = torch.nn.Identity()
    self.i3d.Mixed_4f = torch.nn.Identity()
    self.i3d.MaxPool3d_5a_2x2 = torch.nn.Identity()
    self.i3d.Mixed_5b = torch.nn.Identity()
    self.i3d.Mixed_5c = torch.nn.Identity()
    
    self.conv1 = torch.nn.Sequential(
        torch.nn.Conv3d(in_channels=512, out_channels =512, kernel_size=(1,1,1), stride=(1,1,1)),
        torch.nn.BatchNorm3d(512),
        torch.nn.ReLU()
    )
    self.conv2 = torch.nn.Sequential(
        torch.nn.Conv3d(in_channels=512, out_channels = 256, kernel_size=(1,1,1), stride=(1,1,1)),
        torch.nn.BatchNorm3d(256),
        torch.nn.ReLU()
    )
    self.conv3 = torch.nn.Sequential(
        torch.nn.Conv3d(in_channels=256, out_channels = 64, kernel_size=(1,1,1), stride=(1,1,1)),
        torch.nn.BatchNorm3d(64),
        torch.nn.ReLU()
    )
    self.out = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(64*2*8*8, 128),
    )
  def resize_video(self, video):
    if len(video.shape) == 5:   # b, t, c, h, w 
      b, t, c, h, w = video.shape
      video = video.reshape(b*t, c, h, w)
      video = self.resize(video)
      video = video.reshape(b, t, c, 224, 224)
    else:                   # t, c, h, w
      video = self.resize(video)
    
    return video

  def forward(self, x):

    x = self.resize_video(x)
    x = transpose_channels(x)
    x = x * 2 - 1                 

    x = self.i3d(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.out(x)
    return x