import os
import numpy as np
from tqdm import tqdm

train_dir_path = os.path.join(os.path.dirname(__file__), "dataset", "preprocessed_data", "video_data", "train")
video_file_names = os.listdir(train_dir_path)

current_rgb_mean = np.zeros(3)
current_num_frames = 0

for video_name in tqdm(video_file_names):
    video_path = os.path.join(train_dir_path, video_name)
    video = np.load(video_path).astype(float)

    video /= 255.
    video_rgb_mean = np.mean(video, axis=(0, 1, 2))
    video_num_frames = video.shape[0]

    current_rgb_mean = current_rgb_mean * current_num_frames / (current_num_frames + video_num_frames) + \
                       video_rgb_mean * video_num_frames / (current_num_frames + video_num_frames)
    current_num_frames += video_num_frames


current_rgb_variance = np.zeros(3)
current_num_frames = 0
for video_name in tqdm(video_file_names):
    video_path = os.path.join(train_dir_path, video_name)
    video = np.load(video_path).astype(float)

    video /= 255.
    video_rgb_variance = np.mean((video - current_rgb_mean) ** 2, axis=(0, 1, 2))
    video_num_frames = video.shape[0]

    current_rgb_variance = current_rgb_variance * current_num_frames / (current_num_frames + video_num_frames) + \
                           video_rgb_variance * video_num_frames / (current_num_frames + video_num_frames)
    current_num_frames += video_num_frames

rgb_std = np.sqrt(current_rgb_variance)


np.set_printoptions(precision=16)
print("RGB Mean: ", current_rgb_mean)
print("RGB Std Deviation: ", rgb_std)
