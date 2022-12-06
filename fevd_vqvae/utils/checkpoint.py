import os
import torch
from typing import Tuple
import shutil


class Checkpoint:

    def __init__(self,
                 ckpt_base_dir_path: str,
                 ckpt_cfg_name: str,
                 resume: bool):

        self._checkpoint_path = os.path.join(ckpt_base_dir_path, ckpt_cfg_name)
        path_exists = os.path.exists(self._checkpoint_path)
        if resume:
            assert path_exists, "Cannot resume training. Previous checkpoint do not exist."
        elif path_exists:
            shutil.rmtree(self._checkpoint_path)
        else:
            os.mkdir(self._checkpoint_path)

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
