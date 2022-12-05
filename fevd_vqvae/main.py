import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import torch
import random
from typing import Dict
from fevd_vqvae.utils import Logger, Checkpoint, setup_dataloader
from fevd_vqvae.models import VQModel


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_parser_config() -> Dict:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data_name",
        type=str,
        nargs=1,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "dataset", "preprocessed_data")),
        help="data directory name"
    )

    parser.add_argument(
        "-ckpt",
        "--checkpoint_name",
        type=str,
        nargs=1,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "checkpoints")),
        help="checkpoint directory name"
    )

    parser.add_argument(
        "-l",
        "--log_name",
        type=str,
        nargs=1,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "logs")),
        help="logs directory name"
    )

    parser.add_argument(
        '-cfg',
        '--config_name',
        type=str,
        nargs=1,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "configs", 'baseline.yaml')),
        help="model configuration file name"
    )

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs=1,
        help="resume from the checkpoint in logdir",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="random seed"
    )

    parser.add_argument(
        "-dvc",
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="device for pytorch computations"
    )

    return vars(parser.parse_args())


def get_cfg_file_name(config_file_path: str) -> str:
    config_file_name = os.path.split(config_file_path)[-1]
    cfg_file_name = config_file_name.split('.')[0]
    return cfg_file_name


def verify_logs_ckpt_delete(log_dir_path: str,
                            ckpt_base_dir_path: str,
                            cfg_name: str):
    checkpoint_path = os.path.join(ckpt_base_dir_path, cfg_name)
    log_path = os.path.join(log_dir_path, cfg_name)

    if os.path.exists(checkpoint_path):
        assert os.path.exists(log_path), f"The checkpoints of the {cfg_name} configuration exist, but the logs do not."
        print(f"The logs and the checkpoints for the {cfg_name} will be deleted!!")
        print("Do you want to continue?(y/n)")
        ans = input().lower()
        if ans != 'y':
            exit()
    else:
        assert not os.path.exists(log_path), f"The logs of the {cfg_name} configuration exist, but the checkpoints " \
                                             f"do not."


def main(parser_config: Dict,
         model_cfg_dict: Dict,
         train_cfg_dict: Dict):
    cfg_file_name = get_cfg_file_name(parser_config['config_name'])

    if parser_config['resume'] is not None:
        verify_logs_ckpt_delete(log_dir_path=parser_config['log_name'],
                                ckpt_base_dir_path=parser_config['config_name'],
                                cfg_name=cfg_file_name)

    logger = Logger(log_dir_path=parser_config['log_name'],
                    log_file_name=cfg_file_name,
                    resume=parser_config['resume'])

    checkpoint_logger = Checkpoint(ckpt_base_dir_path=parser_config['checkpoint_name'],
                                   ckpt_cfg_name=cfg_file_name,
                                   resume=parser_config['resume'])

    train_dataloader = setup_dataloader(root_dir_path=parser_config['data_name'], **train_cfg_dict['train_dataloader'])
    eval_dataloaders = setup_dataloader(root_dir_path=parser_config['data_name'], **train_cfg_dict['eval_dataloader'])

    #metrics_trackers = setup_metrics_trackers()

    # model = VQModel(**model_cfg_dict)

    total_steps = int(train_cfg_dict['total_steps'])
    for step in range(1, total_steps+1):
        if step % train_cfg_dict['eval_freq'] == 0:
            # eval()
            #log
            pass
        break



if __name__ == '__main__':
    parser_dict = get_parser_config()
    cfg_dict = OmegaConf.load(parser_dict['config_name'])
    model_cfg_dict = cfg_dict['model']
    train_cfg_dict = cfg_dict['setup']

    seed_everything(parser_dict['seed'])
    main(parser_dict, model_cfg_dict, train_cfg_dict)
