import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm import tqdm
import gc
import torch.optim as optim
try:
    from fevd_vqvae.models.temporal_net import TemporalModel
except ModuleNotFoundError:
    from models.temporal_net import TemporalModel
try:
    from fevd_vqvae.utils.dataset import VideoPermutationDataset
except ModuleNotFoundError:
    from utils.dataset import VideoPermutationDataset
except ImportError:
    from utils.dataset import VideoPermutationDataset
from omegaconf import OmegaConf
import os
from fevd_vqvae.utils import Logger


def get_parser_config():
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
        "-i3dckpt",
        "--i3dckpt",
        type=str,
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "rgb_imagenet.pt")),
        help="Path to pretrained i3d checkpoint"
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
        default=os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "configs", 'train_temporal.yaml')),
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

    return vars(parser.parse_args())

def seed_everything(seed: int):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = get_parser_config()
    seed_everything(args['seed'])
    train_cfg = OmegaConf.load(args['config_name'])
    print(train_cfg)
    train_data = VideoPermutationDataset(**train_cfg.data.train, root_dir_path=args['data_name'])
    train_loader = DataLoader(train_data,**train_cfg.dataloader.train)
    
    val_data = VideoPermutationDataset(**train_cfg.data.val, root_dir_path=args['data_name'])
    val_loader = DataLoader(val_data,**train_cfg.dataloader.val)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    if "params" not in train_cfg.model:
        test = TemporalModel(i3d_ckpt=args['i3dckpt']).to(device)
    else:
        test = TemporalModel(**train_cfg.model.params, i3d_ckpt=args['i3dckpt']).to(device)
    opt = optim.Adam(test.parameters(), **train_cfg.opt.params)
    criterion = nn.TripletMarginLoss(reduction='none')      # We can use PyTorch's Triplet Loss function
    nepochs = train_cfg.training.epochs
    log_steps = train_cfg.training.get('log_steps', 100)
    
    save_ckpt_dir = os.path.join(args['checkpoint_name'], "temporal_net")
    os.makedirs(save_ckpt_dir, exist_ok=True)
    
    logger = Logger(log_dir_path=args['log_name'],
                    log_file_name="temporal_net",
                    resume=args['resume'])
    
    print("Beginning training loop...")
    global_step = 0
    for epoch in range(nepochs):
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
            
            if m % log_steps == 0:
                print(f" Training Loss: {loss}")
                logger._log_scalar(float(loss), "train/loss", global_step)
                test.eval()
                #print("Validation loss: ")
                for i, v in tqdm(enumerate(val_loader)):
                    with torch.no_grad():
                        anchor, positive, negative = v
                        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                        anchor_embs, positive_embs, negative_embs = test(anchor), test(positive), test(negative)
                        loss_per_instance = criterion(anchor_embs, positive_embs, negative_embs)
                        non_zero_losses = torch.sum(loss_per_instance != 0)

                    if non_zero_losses > 0:
                        val_loss = torch.sum(loss_per_instance) / (non_zero_losses)        # average loss
                        if type(val_loss) != float: val_loss = float(val_loss.item())
                        print("Validation loss: ", val_loss)
                        logger._log_scalar(val_loss, "val/loss", global_step)
                #Reset model back to training for next step
                test.train()       
            global_step+=1          
        torch.save({'epoch': epoch,
            'model_state_dict': test.state_dict(),
            'opt_state_dict': opt.state_dict(),
            }, os.path.join(save_ckpt_dir ,f"temporal_net_e{epoch}.pt"))