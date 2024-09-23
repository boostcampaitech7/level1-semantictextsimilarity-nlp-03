import os
import torch
import numpy as np
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from module.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from utils import *
import hydra
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(config):
    config.pwd = os.getcwd()
    config.wandb.enable = True
    with wandb.init():
        wandb_config = wandb.config
        config.data_module.args.batch_size = wandb_config.batch_size
        config.optimizer.args.lr = wandb_config.learning_rate

        config = OmegaConf.to_container(config, resolve=True)

        # 1. set data_module(=pl.DataModule class)
        data_module = init_obj(
            config["data_module"]["type"], config["data_module"]["args"], module_data
        )

        # 2. set model(=nn.Module class)
        model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)

        # 3. set deivce(cpu or gpu)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = model.to(device)

        # 4. set loss function & matrics
        criterion = getattr(module_loss, config["loss"])
        metrics = [getattr(module_metric, met) for met in config["metrics"]]

        # 5. set optimizer & learning scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = init_obj(
            config["optimizer"]["type"],
            config["optimizer"]["args"],
            torch.optim,
            trainable_params,
        )
        lr_scheduler = init_obj(
            config["lr_scheduler"]["type"],
            config["lr_scheduler"]["args"],
            torch.optim.lr_scheduler,
            optimizer,
        )

        trainer = Trainer(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            device=device,
            data_module=data_module,
            lr_scheduler=lr_scheduler,
        )

        # 6. train
        trainer.train()

if __name__ == "__main__":
    sweep_config = {
        'method': 'random',  # 또는 'bayes'
        'metric': {
            'name': 'val_pearson',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'uniform',  # 또는 'log_uniform'
                'min': 0.00001,
                'max': 0.0001
            },
            'batch_size': {
                'values': [16, 32]
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="STS")

    wandb.agent(sweep_id, function=main, count=10)