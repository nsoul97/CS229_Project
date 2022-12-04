import os
import torch
from typing import Tuple

CKPT_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "checkpoints"))


class Checkpoint:

    def __init__(self,
                 config: dict):

        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)

        checkpoint_run_name = '__'.join([f'{k}:{v}' for k, v in config.items()])
        self._checkpoint_run_path = os.path.join(CKPT_DIR, checkpoint_run_name)

        if not os.path.exists(self._checkpoint_run_path):
            os.mkdir(self._checkpoint_run_path)

    def save_checkpoint(self,
                        config: dict,
                        model_state_dict: dict,
                        step: int):

        checkpoint_filepath = os.path.join(self._checkpoint_run_path, f'{step}.ckpt')
        torch.save({'config': config,
                    'model_state_dict': model_state_dict},
                   checkpoint_filepath)

    def load_checkpoint(self,
                        step: int) -> Tuple[dict, dict]:

        checkpoint_filepath = os.path.join(self._checkpoint_run_path, f'{step}.ckpt')
        config, model_state_dict = torch.load(checkpoint_filepath)
        return config, model_state_dict
