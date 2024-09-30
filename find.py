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
        # ---------------------------------Connect----------------------------------------
        config.data_module.args.batch_size = wandb_config.batch_size
        config.optimizer.args.lr = wandb_config.learning_rate
        config.optimizer.args.weight_decay = wandb_config.weight_decay
        # config.arch.args.dropout_rate = wandb_config.dropout_rate
        config.arch.args.lora_r = wandb_config.lora_r
        config.arch.args.lora_alpha = wandb_config.lora_alpha
        config.arch.args.lora_dropout = wandb_config.lora_dropout
        # --------------------------------------------------------------------------------
        config = OmegaConf.to_container(config, resolve=True)

        # 1. set data_module(=pl.DataModule class)
        data_module = init_obj(
            config["data_module"]["type"], config["data_module"]["args"], module_data
        )

        # 2. set model(=nn.Module class)
        model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)
        model.plm.config.pad_token_id = data_module.tokenizer.pad_token_id

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

        # 6. print model summary, 학습 가능한 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params_count}")
        print(model)

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
        "name": "SLMSTS",
        "method": "random",  # 또는 'bayes'
        "metric": {"name": "val_pearson", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "uniform",  # 또는 'log_uniform'
                "min": 0.000001,
                "max": 0.00005,
            },
            "batch_size": {"values": [32, 64]},  # 모델에 따라 다르게 설정
            "dropout_rate": {"values": [0.1, 0.2, 0.3]},

            # 'optimizer': {
            #     'values': ['adam', 'adamw']
            # },
            # 'lr_scheduler': {
            #     'values': [50, 100]
            # },

            "weight_decay": {
                "distribution": "uniform",
                "min": 0.000001,
                "max": 0.00005,
            },
            "lora_r": {"values": [32, 64]},
            "lora_alpha": {"values": [8, 16, 32]},
            "lora_dropout": {"values": [0.05, 0.1, 0.2]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="STS")

    wandb.agent(sweep_id, function=main, count=15)
