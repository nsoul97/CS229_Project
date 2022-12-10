# Learning to compress and reconstruct video frames for two-stage video prediction

#### As part of the final course project for CS229: Machine Learning, Fall 2022
#### Authors: [Nikos Soulounias](https://github.com/nsoul97), [Andrew Cheng](https://github.com/acheng416)


## Introduction
Recently, Transformer-based architectures have achieved many state-of-the-art results in Computer Vision, NLP, and other fields, 
at the cost of quadratic time and memory complexity. In the subdomain of video prediction, this heavy computational cost is even more restrictive, as videos
are often composed of many high-resolution frames, leading to large interest in compression-reconstruction models. 


In this project, we focus on the first stage of the video prediction approaches that can be summarized by two core objectives: 
- Learning an **Encoder** that maps input RGB frames to a downsampled latent space
- Learning a **Decoder** that reconstructs the inputs from the latent space encodings. 

## Requirements
The code in this repo was run using Python 3.8.0+ and installed using the latest version of pip using the following
(a virtual environment is highly recommended):

```
cd CS229_Project/ && pip install .
```
A detailed list of required packages can be found in the *requirements.txt* file, including: 
#### numpy, scipy, torch, torchvision, einops, tqdm, omegaconf, gdown, h5py, pandas, opencv-python, imageio, moviepy, tensorboard

## Dataset
We choose the large-scale robot learning [RoboNet](https://arxiv.org/pdf/1910.11215.pdf) dataset, which contains multiple camera angle
videos of 7 different robots and over 15 million frames in the full dataset. The dataset can be downloaded by following the 
instructions in the official repository [here](https://github.com/SudeepDasari/RoboNet/wiki/Getting-Started).

After downloading and extracting the full RoboNet dataset, you will need to the root of this repo and move the **hdf5/** folder containing raw data to the following directory. 
Then, run the necessary preprocessing scripts (Warning time-consuming!):
```
mv PATH_TO_HDF5/ ./data/dataset/ \
 python ./data/preprocess_robonet.py
```
Next, for model training, use the following:
```
python fevd-vqvae/main.py -d PATH_TO_PREPROCESSED_DATA -cfg configs/YOUR_CONFIG.yaml 
```
## Baseline
We use a [VQ-VAE](https://papers.nips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf) model, 
which encodes the RGB frames and decodes the latents in the spatial domain. 


The code in this branch is inspired by and or directly references parts of [taming-transformers](https://github.com/CompVis/taming-transformers) 
for the VQ-VAE and VQ-GAN base code, [RoboNet](https://github.com/SudeepDasari/RoboNet) for preprocessing code snippets, 
and [fvd-comparison](https://github.com/universome/fvd-comparison) for the FVD implementation.


