import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import torch
import random
from typing import Dict


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
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data")),
        help="data directory name"
    )

    parser.add_argument(
        "-ckpt",
        "--ckpt_name",
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
        '-mc',
        '--model_config_name',
        type=str,
        nargs=1,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "model_configs", '2d_vqgan.yaml')),
        help="model configuration file name"
    )

    parser.add_argument(
        '-dc',
        '--data_config_name',
        type=str,
        nargs=1,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data_configs", 'robonet.yaml')),
        help="dataset configuration file name"
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

    return vars(parser.parse_args())


def main(parser_config: Dict,
         model_config: Dict,
         data_config: Dict):
    return None


if __name__ == '__main__':
    parser_config = get_parser_config()
    model_config = OmegaConf.load(parser_config['model_config_name'])
    data_config = OmegaConf.load(parser_config['data_config_name'])

    seed_everything(parser_config['seed'])
    main(parser_config, model_config, data_config)
