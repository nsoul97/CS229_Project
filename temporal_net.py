import torch
import torch.nn as nn
import torch.utils.data as data
from dataset import unnormalize
from torch.utils.data import DataLoader
import random
import pandas as pd
import numpy as np
import os
import torchvision
from torchvision import datasets
import torchvision.transforms as T

import videotransforms
from torchinfo import summary
from pytorch_i3d import InceptionI3d
from torchvision import transforms
from tqdm import tqdm
import gc
import torch.optim as optim

def transpose_channels(video):
  if len(video.shape) == 5:   # b, t, c, h, w  ->  b, c, t, h, w
    video = video.transpose(1, 2)
  else:                   # t, c, h, w -> c, t, h, w
    video = video.transpose(0, 1)
  return video

class TemporalModel(nn.Module):
  def __init__(self, num_classes=400, spatial_squeeze=True, final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
    super(TemporalModel, self).__init__()
    
    self.resize = transforms.Resize((224, 224))

    self.i3d = InceptionI3d(400, in_channels=3)
    self.i3d.load_state_dict(torch.load('rgb_imagenet.pt'))
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

class RandomizePixels:
    def __init__(self, pixel_distort_prob=0.005):
        self._pixel_distort_prob = pixel_distort_prob

    def __call__(self, frames):
        t, c, h, w = frames.shape
        pixel_p = torch.full((t, c, h, w), fill_value=self._pixel_distort_prob)
        pixel_distort_bools = torch.bernoulli(pixel_p).to(bool)   
        noise = torch.rand((t, c, h, w))
        frames[pixel_distort_bools] = noise[pixel_distort_bools]
        return frames



class VideoPermutationDataset(data.Dataset):
  def __init__(self, augment=True, split = "train"):
      super(VideoPermutationDataset, self).__init__()
      #self._robonet_video_dataset = robonet_video_dataset
      self.split = split
      df_path = f"./{split}.csv"
      self.T = 12
      self.augment = augment

      if augment:
        self.gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=3)
        self.random_pixels = RandomizePixels()
      else:
        self.gaussian_blur = None
        self.random_pixels = None
      self.video_files = pd.read_csv(df_path)['video_files'].values
      
      metadata_file_path = "./metadata.csv"
      metadata = pd.read_csv(metadata_file_path, index_col=0, low_memory=False)
      traj_frames = metadata['state_T']

      self._data = []
      for video_fname in self.video_files:
          traj_name = '_'.join(video_fname.split('_')[:-1]) + '.hdf5'
          video_frames = traj_frames.loc[traj_name]
          video_num_windows = video_frames - 12 + 1
          self._data += list(zip([video_fname] * video_num_windows, list(range(video_num_windows))))

      self._total_frames = 12

  def __len__(self):
    return len(self.video_files)

  def transform(self, frames):
    transforms  = []
    if random.uniform(0, 1) < 0.8:
      if random.uniform(0, 1) < 0.75:    # apply randomize_pixel transfrom
        transforms.append(self.random_pixels)
      if random.uniform(0, 1) < 0.2:     # apply gaussian blur transform
        transforms.append(self.random_pixels)
      
      random.shuffle(transforms)    # randomize the order of the transformations to be applied
    
      for t in transforms:
        frames = t(frames)
    
    return frames

  def __getitem__(self, idx):
    file_name, start_frame_idx = self._data[idx]
    file_path = os.path.join(f"/mnt/d/dataset/preprocessed_data/video_data/{self.split}", file_name)
    video = np.load(file_path)
    video_segment = torch.from_numpy(video[start_frame_idx: start_frame_idx + self._total_frames])
    video_segment = torch.permute(video_segment, [0, 3, 1, 2]).to(memory_format=torch.contiguous_format).float()
    #video_segment /= 255.
    #video_segment = normalize(video_segment)

    anchor = video_segment
    anchor = unnormalize(anchor)                 # We were normalizing with RoboNet statistics, but the i3d model is trained without normalizing the inputs.

    achor_ind = torch.arange(0, self.T, dtype=torch.int)

    perm_ind1 = torch.randperm(self.T)
    shift_dist1 = torch.sum(torch.abs(achor_ind - perm_ind1))

    shift_dist2 = shift_dist1
    while shift_dist1 == shift_dist2:
      perm_ind2 = torch.randperm(self.T)
      shift_dist2 = torch.sum(torch.abs(achor_ind - perm_ind2))

    if shift_dist1 > shift_dist2:
      negative = anchor[perm_ind1, ...]
      positive = anchor[perm_ind2, ...]
    else:
      negative = anchor[perm_ind2, ...]
      positive = anchor[perm_ind1, ...]

    if self.augment:
      positive = self.transform(positive)
      negative = self.transform(negative)     

    return anchor, positive, negative


if __name__ == '__main__':
    def seed_everything(seed: int):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
    seed_everything(23)
    train_data = VideoPermutationDataset(augment=False)
    train_loader = DataLoader(train_data,num_workers=4, batch_size=100, shuffle=True, drop_last=True)
    
    val_data = VideoPermutationDataset(augment=False, split="val")
    val_loader = DataLoader(val_data,num_workers=1, batch_size=100, shuffle=False, drop_last=False)
    
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test = TemporalModel().to(device)
    opt = optim.Adam(test.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(reduction='none')      # We can use PyTorch's Triplet Loss function

    total_steps = 100000
    #pbar = tqdm(total=total_steps)
    step_loss = 0.0
    for epoch in range(10):
      test.train()
      for m, b in tqdm(enumerate(train_loader)):
        opt.zero_grad()
        anchor, positive, negative = b
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_embs, positive_embs, negative_embs = test(anchor), test(positive), test(negative)
        loss_per_instance = criterion(anchor_embs, positive_embs, negative_embs)
        non_zero_losses = torch.sum(loss_per_instance != 0)

        if non_zero_losses > 0:
          loss = torch.sum(loss_per_instance) / (non_zero_losses)        # average loss
          loss = loss #/ grad_accum_steps                               # accumulate gradients
          loss.backward()
          opt.step()
        #print("-"*20)
        if m % 100 == 0:
          print(f"Loss: {loss}")
      torch.save({'epoch': epoch,
        'model_state_dict': test.state_dict(),
        'opt_state_dict': opt.state_dict(),
        }, f"temporal_net_e{epoch}.pt")
      test.eval()
      print("Computing validation loss: ")
      for i, v in tqdm(enumerate(val_loader)):
        anchor, positive, negative = v
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_embs, positive_embs, negative_embs = test(anchor), test(positive), test(negative)
        loss_per_instance = criterion(anchor_embs, positive_embs, negative_embs)
        non_zero_losses = torch.sum(loss_per_instance != 0)

        if non_zero_losses > 0:
          loss = torch.sum(loss_per_instance) / (non_zero_losses)        # average loss
          loss = loss #/ grad_accum_steps 
          print("Validation loss: ", loss)                    