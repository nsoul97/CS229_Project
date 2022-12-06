import os
import torch
import shutil
from typing import Union
from queue import Queue

class Checkpoint:
    def __init__(self,
                 ckpt_base_dir_path: str,
                 ckpt_cfg_name: str,
                 resume: bool,
                 num_last_checkpoints: int):

        self._checkpoint_log = {'total_loss': float('inf'),
                                'FVD': float('inf'),
                                'PSNR': float('-inf'),
                                'SSIM': float('-inf'),
                                'LPIPS': float('inf')}

        self._checkpoint_path = os.path.join(ckpt_base_dir_path, ckpt_cfg_name)
        path_exists = os.path.exists(self._checkpoint_path)
        if resume:
            assert path_exists, "Cannot resume training. Previous checkpoint do not exist."
        elif path_exists:
            shutil.rmtree(self._checkpoint_path)
            path_exists = False

        if not path_exists:
            os.mkdir(self._checkpoint_path)
        self._checkpoints_queue = Queue(maxsize=num_last_checkpoints)

    def save_checkpoint(self,
                        model_state_dict: dict,
                        opt_state_dict: dict,
                        val_log: dict,
                        step: int):

        if self._checkpoints_queue.full():
            last_checkpoint = self._checkpoints_queue.get()
            os.remove(last_checkpoint)
        new_checkpoint_name = os.path.join(self._checkpoint_path, f'{step}.ckpt')
        self._checkpoints_queue.put(new_checkpoint_name)

        checkpoints = [f'{step}.ckpt']
        if val_log['total_loss'] < self._checkpoint_log['total_loss']:
            checkpoints.append('lowest_loss.ckpt')
            self._checkpoint_log['total_loss'] = val_log['total_loss']

        if val_log['FVD'] < self._checkpoint_log['FVD']:
            checkpoints.append('lowest_fvd.ckpt')
            self._checkpoint_log['FVD'] = val_log['FVD']

        if val_log['PSNR'] > self._checkpoint_log['PSNR']:
            checkpoints.append('highest_psnr.ckpt')
            self._checkpoint_log['PSNR'] = val_log['PSNR']

        if val_log['SSIM'] > self._checkpoint_log['SSIM']:
            checkpoints.append('highest_ssim.ckpt')
            self._checkpoint_log['SSIM'] = val_log['SSIM']

        if val_log['LPIPS'] < self._checkpoint_log['LPIPS']:
            checkpoints.append('lowest_lpips.ckpt')
            self._checkpoint_log['LPIPS'] = val_log['LPIPS']

        for checkpoint_file_name in checkpoints:
            checkpoint_filepath = os.path.join(self._checkpoint_path, checkpoint_file_name)
            torch.save({'step': step,
                        'model_state_dict': model_state_dict,
                        'opt_state_dict': opt_state_dict,
                        'checkpoint_log': self._checkpoint_log}, checkpoint_filepath)

    def _load_queue(self, step):
        checkpoint_names = [ch_file_name.split('.')[0] for ch_file_name in os.listdir(self._checkpoint_path)]
        checkpoint_steps = sorted([int(ch_name) for ch_name in checkpoint_names if ch_name.isnumeric()])
        print(checkpoint_steps)

        for ch_step in checkpoint_steps:
            ch_path = os.path.join(self._checkpoint_path, f'{ch_step}.ckpt')
            if ch_step <= step:
                self._checkpoints_queue.put(ch_path)
            else:
                os.remove(ch_path)

    def load_checkpoint(self,
                        checkpoint_name: Union[str, None]) -> dict:

        if checkpoint_name is not None:
            print(self._checkpoint_path, checkpoint_name)
            checkpoint_filepath = os.path.join(self._checkpoint_path, checkpoint_name)
            checkpoint_state_dict = torch.load(checkpoint_filepath)
            self._checkpoint_log = checkpoint_state_dict['checkpoint_log']
            self._load_queue(checkpoint_state_dict["step"])

            print(f"Loading checkpoint from {checkpoint_filepath}")
            print("Best statistics so far: ", self._checkpoint_log)
            print("Starting after step ", checkpoint_state_dict["step"])
            return checkpoint_state_dict
        else:
            return {'step': 0,
                    'model_state_dict': None,
                    'opt_state_dict': None}
