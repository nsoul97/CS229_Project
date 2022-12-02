import argparse
import os
import h5py
import numpy as np
from robonet_utils import load_metadata, load_camera_imgs, MetaDataContainer
from PIL import Image
import shutil
from typing import Dict
import multiprocessing as mp
import random
from tqdm import tqdm


def get_parser_config() -> Dict:
    parser = argparse.ArgumentParser("Preprocess the RoboNet dataset to speed up the training")

    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=64,
        help="The (H,W) video frames will be rescaled to (D, D)."
    )

    parser.add_argument(
        "-t",
        "--min_video_frames",
        type=int,
        default=12,
        help="The required minimum frames to include a video in the dataset."
    )

    parser.add_argument(
        "-l",
        "--load_dir_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "hdf5"),
        help="The path of the raw hdf5 data."
    )

    parser.add_argument(
        "-s",
        "--save_dir_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset", "preprocessed_data"),
        help="The path where the preprocessed data will be saved."
    )

    parser.add_argument(
        "-ntrj",
        "--num_trajectories",
        type=int,
        default=None,
        help="The number of trajectories that will be used."
    )

    parser.add_argument(
        "-nv",
        "--n_val",
        type=int,
        default=256,
        help="The validation set consists of n_val videos."
    )

    parser.add_argument(
        "-nts",
        "--n_test",
        type=int,
        nargs=1,
        default=256,
        help="The test set consists of n_test videos."
    )

    parser.add_argument(
        "--debug",
        action='store_true',
        help="Save some examples to visualize everything is correct."
    )

    parser.add_argument(
        "-r",
        "--random_seed",
        type=int,
        default=23,
        help="The random seed used to split the dataset into training, validation and test sets."
    )

    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=mp.cpu_count() - 1,
        help="The number of workers that are used to parallelize the preprocessing."
    )

    return vars(parser.parse_args())


def process_hdf5(files,
                 data_dir_path,
                 data,
                 dim,
                 load_folder,
                 example_dir_path,
                 debug,
                 tqdm_queue):
    if debug:
        flags = np.zeros(4, dtype=bool)

    target_dims = [dim, dim]

    for i in range(len(files)):
        f_path = files[i]

        sample_name = f_path.split(f"{load_folder}/")[-1]
        name = sample_name.split('.hdf5')[0]
        file_metadata = data.get_file_metadata(sample_name)
        ncam, img_encoding, img_fmt, frames = file_metadata['ncam'], file_metadata['img_encoding'], file_metadata[
            'image_format'], file_metadata['img_T']

        with h5py.File(f_path, "r") as file_pointer:
            for cam_index in range(file_metadata['ncam']):
                sample = load_camera_imgs(cam_index, file_pointer, file_metadata, target_dims)

                video_data_file_path = os.path.join(data_dir_path, f"{name}_{cam_index}.npy")
                with open(video_data_file_path, 'wb') as wf:
                    np.save(wf, sample)

            if debug:  # Save an example of processed image
                save = False
                if img_encoding == 'mp4' and img_fmt == 'BGR' and (not flags[0]):
                    save = True
                    flags[0] = 1
                elif img_encoding == 'jpg' and img_fmt == 'BGR' and (not flags[1]):
                    save = True
                    flags[1] = 1
                elif img_encoding == 'mp4' and img_fmt == 'RGB' and (not flags[2]):
                    save = True
                    flags[2] = 1
                elif img_encoding == 'jpg' and img_fmt == 'RGB' and (not flags[3]):
                    save = True
                    flags[3] = 1

                if save:
                    example_img_file_path = os.path.join(example_dir_path,
                                                         f'{img_encoding}_{img_fmt}_{name}_{cam_index}.png')
                    Image.fromarray(sample[0, :, :, :]).save(example_img_file_path)

        tqdm_queue.put(1, block=True)


def preprocess(data: MetaDataContainer,
               dim: int,
               load_folder: str,
               save_folder: str,
               n_traj: int,
               num_workers: int,
               debug: bool):
    """
    Args:
      - data (list): list of hdf5 file paths to process
      - dim (int): the target dimension
      - load_folder (str): folder containing the hdf5 files
      - save_folder (str): parent folder to create the video_data folder and the metadata.csv file
    """

    data_dir_path = os.path.join(save_folder, "video_data")
    example_dir_path = os.path.join(save_folder, "examples")
    md_file_path = os.path.join(save_folder, "metadata.csv")

    if os.path.exists(data_dir_path): shutil.rmtree(data_dir_path)
    os.makedirs(data_dir_path)

    if os.path.exists(example_dir_path): shutil.rmtree(example_dir_path)
    if debug: os.makedirs(example_dir_path)

    n_traj = n_traj or len(data)

    files = data.files
    df = data.frame
    if n_traj < len(data):
        random.shuffle(files)
        files = files[:n_traj]
        df = data.frame.loc[[fname.split('/')[-1] for fname in files]]
    df.to_csv(md_file_path)

    start_idx = 0
    worker_num_files = len(files) // num_workers
    processes = []
    tqdm_queue = mp.Queue()
    for i in range(num_workers):
        end_idx = start_idx + worker_num_files if i < num_workers - 1 else len(files)
        worker_files = files[start_idx: end_idx]
        p = mp.Process(target=process_hdf5,
                       args=(worker_files, data_dir_path, data, dim, load_folder, example_dir_path,
                             debug, tqdm_queue))
        p.start()
        start_idx += worker_num_files
        processes.append(p)

    files_processed = 0
    tqdm_update_files = 0
    bar = tqdm(total=len(files))
    while files_processed < len(files):
        new_files_processed = tqdm_queue.get(block=True)
        tqdm_update_files += new_files_processed
        files_processed += new_files_processed

        if tqdm_update_files == 10 or files_processed == len(files):
            bar.update(tqdm_update_files)
            tqdm_update_files = 0

    for p in processes:
        p.join()


def split_dataset(save_folder, n_val, n_test):
    data_dir_path = os.path.join(save_folder, "video_data")
    train_data_dir_path = os.path.join(data_dir_path, "train")
    val_data_dir_path = os.path.join(data_dir_path, "val")
    test_data_dir_path = os.path.join(data_dir_path, "test")

    if os.path.exists(train_data_dir_path): shutil.rmtree(train_data_dir_path)
    if os.path.exists(val_data_dir_path): shutil.rmtree(val_data_dir_path)
    if os.path.exists(test_data_dir_path): shutil.rmtree(test_data_dir_path)

    os.makedirs(train_data_dir_path)
    os.makedirs(val_data_dir_path)
    os.makedirs(test_data_dir_path)

    video_files = [vid_fname for vid_fname in os.listdir(data_dir_path) if vid_fname.split('.')[-1] == 'npy']
    n_train = len(video_files) - parser_config['n_val'] - parser_config['n_test']

    random.shuffle(video_files)
    for i, video_fname in enumerate(video_files):

        old_video_path = os.path.join(data_dir_path, video_fname)
        if i < n_train:
            new_video_path = os.path.join(train_data_dir_path, video_fname)
        elif i < n_train + n_val:
            new_video_path = os.path.join(val_data_dir_path, video_fname)
        else:
            new_video_path = os.path.join(test_data_dir_path, video_fname)
        os.replace(old_video_path, new_video_path)


if __name__ == '__main__':
    parser_config = get_parser_config()
    random.seed(parser_config['random_seed'])

    robonet_data = load_metadata('data/dataset/hdf5')
    preprocess(data=robonet_data,
               dim=parser_config['dim'],
               load_folder=parser_config['load_dir_path'],
               save_folder=parser_config['save_dir_path'],
               n_traj=parser_config['num_trajectories'],
               num_workers=parser_config['num_workers'],
               debug=parser_config['debug'])

    split_dataset(save_folder=parser_config['save_dir_path'],
                  n_val=parser_config['n_val'],
                  n_test=parser_config['n_test'])
