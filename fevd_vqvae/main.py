import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import torch
import random
from typing import Dict
from fevd_vqvae.utils import Logger, Checkpoint, setup_dataloader
from fevd_vqvae.models import VQModel, LossTracker
from fevd_vqvae.metrics import MetricsTracker
import torch.utils.data as data


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
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "data", "dataset", "preprocessed_data")),
        help="data directory name"
    )

    parser.add_argument(
        "-ckpt",
        "--checkpoint_name",
        type=str,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "checkpoints")),
        help="checkpoint directory name"
    )

    parser.add_argument(
        "-l",
        "--log_name",
        type=str,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "logs")),
        help="logs directory name"
    )

    parser.add_argument(
        '-cfg',
        '--config_name',
        type=str,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "configs", 'baseline.yaml')),
        help="model configuration file name"
    )

    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
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

    print(checkpoint_path)
    if os.path.exists(checkpoint_path):
        assert os.path.exists(log_path), f"The checkpoints of the {cfg_name} configuration exist, but the logs do not."
        print(f"The logs and the checkpoints for the {cfg_name} will be deleted!!")
        print("Do you want to continue?(y/n)")
        ans = input().lower()
        if ans != 'y':
            print("Exiting...")
            exit()
    else:
        assert not os.path.exists(log_path), f"The logs of the {cfg_name} configuration exist, but the checkpoints " \
                                             f"do not."


def eval(dataloader: data.DataLoader,
         model: VQModel,
         split: str,
         logger: Logger,
         device: torch.device,
         step: int):
    metrics_tracker = MetricsTracker(lpips_model=model.loss.perceptual_loss_2d,
                                     inception_i3d_model=model.loss.inception_i3d,
                                     batch_size=len(dataloader.dataset),
                                     device=device)

    loss_tracker = LossTracker(total_steps=len(dataloader))

    for video_inputs in dataloader:
        video_inputs = video_inputs.to(device)
        with torch.inference_mode():
            video_reconstructions, _, loss_log = model.step(video_inputs)
        metrics_tracker.update(video_inputs, video_reconstructions)
        loss_tracker.update(loss_log)

    metrics_log = metrics_tracker.compute()
    logger.log_dict(metrics_log, split, step)

    loss_log = loss_tracker.compute()
    logger.log_dict(loss_log, split, step)

    eval_log = dict()
    for k, v in metrics_log.items():
        eval_log[k] = v
    eval_log['total_loss'] = loss_log['total_loss']
    return eval_log


def select_and_visualize_examples(model: VQModel,
                                  dataloader: data.DataLoader,
                                  split: str,
                                  logger: Logger,
                                  step: int,
                                  imgs_per_grid: int,
                                  total_imgs: int,
                                  num_videos: int,
                                  device: torch.device) -> None:
    real_videos = next(iter(dataloader)).to(device)
    with torch.inference_mode():
        rec_videos, _ = model(real_videos)

    save_real_videos = real_videos[:num_videos]
    save_rec_videos = rec_videos[:num_videos]

    save_real_videos_imgs = real_videos[:total_imgs]
    save_rec_videos_imgs = rec_videos[:total_imgs]

    rand_img_ind = torch.randint(low=0,
                                 high=real_videos.shape[1] - 1,
                                 size=(total_imgs,))

    save_real_imgs = torch.stack([save_real_videos_imgs[vid_ind, img_ind] for vid_ind, img_ind in enumerate(rand_img_ind)], dim=0)
    save_rec_imgs = torch.stack([save_rec_videos_imgs[vid_ind, img_ind] for vid_ind, img_ind in enumerate(rand_img_ind)], dim=0)

    logger.log_visualizations(split=split,
                              input_videos=save_real_videos, reconstruction_videos=save_rec_videos,
                              input_imgs=save_real_imgs, reconstruction_imgs=save_rec_imgs,
                              step=step, nrow=imgs_per_grid)


def main(parser_config: Dict,
         model_cfg_dict: Dict,
         train_cfg_dict: Dict):
    cfg_file_name = get_cfg_file_name(parser_config['config_name'])

    if parser_config['resume'] is None:
        verify_logs_ckpt_delete(log_dir_path=parser_config['log_name'],
                                ckpt_base_dir_path=parser_config['checkpoint_name'],
                                cfg_name=cfg_file_name)

    logger = Logger(log_dir_path=parser_config['log_name'],
                    log_file_name=cfg_file_name,
                    resume=parser_config['resume'])

    checkpoint_logger = Checkpoint(ckpt_base_dir_path=parser_config['checkpoint_name'],
                                   ckpt_cfg_name=cfg_file_name,
                                   resume=parser_config['resume'],
                                   **train_cfg_dict['checkpoint'])
    ckpt_dict = checkpoint_logger.load_checkpoint(parser_config['resume'])

    device = torch.device(parser_config['device'])
    model = VQModel(**model_cfg_dict, ckpt_sd=ckpt_dict['model_state_dict']).to(device)
    opt = model.configure_optimizer(train_cfg_dict['learning_rate'], opt_sd=ckpt_dict['opt_state_dict'])  # get optimizer for VQModel

    train_dataloader = setup_dataloader(root_dir_path=parser_config['data_name'], **train_cfg_dict['train_dataloader'])
    train_loss_tracker = LossTracker(total_steps=train_cfg_dict['grad_updates_per_step'])

    eval_dataloaders = setup_dataloader(root_dir_path=parser_config['data_name'], **train_cfg_dict['eval_dataloader'])
    for split in train_cfg_dict['eval_splits']:
        if len(eval_dataloaders[split].dataset) % eval_dataloaders[split].batch_size != 0:
            raise ValueError(f"The batch size of the {split} DataLoader must divide the length of the "
                             f"dataset {len(eval_dataloaders[split].dataset)}")

    train_dataloader_iter = iter(train_dataloader)
    total_steps = int(train_cfg_dict['total_steps'])
    step = ckpt_dict['step']
    small_step = 0
    while step < total_steps:  # Start steps

        try:
            x = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            x = next(train_dataloader_iter)

        x = x.to(device)

        _, loss, loss_log = model.step(x)  # train step
        train_loss_tracker.update(loss_log)
        loss = loss / train_cfg_dict['grad_updates_per_step']  # normalize the loss for gradient accumulation
        loss.backward()  # backward

        small_step += 1
        if small_step == train_cfg_dict['grad_updates_per_step']:
            opt.step()  # update parameters
            opt.zero_grad()  # clear gradients
            small_step = 0
            step += 1
            step_loss_dict = train_loss_tracker.compute()
            logger.log_dict(log_dict=step_loss_dict, split='train', step=step)

            if step % train_cfg_dict['eval_freq'] == 0:
                print(f"Step {step}: Evaluating...")
                model.eval()

                for split, dataloader in eval_dataloaders.items():
                    select_and_visualize_examples(model=model,
                                                  dataloader=dataloader,
                                                  split=split,
                                                  logger=logger,
                                                  step=step,
                                                  device=device,
                                                  **train_cfg_dict['logging'])

                for split in train_cfg_dict['eval_splits']:
                    log = eval(dataloader=eval_dataloaders[split],
                               model=model,
                               logger=logger,
                               split=split,
                               step=step,
                               device=device)

                    if split == 'val': val_log = log

                checkpoint_logger.save_checkpoint(model_state_dict=model.state_dict(),
                                                  opt_state_dict=opt.state_dict(),
                                                  val_log=val_log,
                                                  step=step)

                model.train()


if __name__ == '__main__':
    parser_dict = get_parser_config()
    cfg_dict = OmegaConf.load(parser_dict['config_name'])
    model_cfg_dict = cfg_dict['model']
    train_cfg_dict = cfg_dict['setup']

    seed_everything(parser_dict['seed'])
    main(parser_dict, model_cfg_dict, train_cfg_dict)
