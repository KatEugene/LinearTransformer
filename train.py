import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.globals import ROOT_PATH
from src.utils.helpers import set_random_seed
from src.utils.preprocessing import get_dataloaders
from src.trainer.trainer import Trainer


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(config):
    set_random_seed(config.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders, tokenizer = get_dataloaders(config)
    model = instantiate(config.model, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer).to(device)
    criterion = instantiate(config.loss_function, ignore_index=tokenizer.pad_token_id)
    metrics = instantiate(config.metrics)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, epoch_len=len(dataloaders["train"]))
    transforms = instantiate(config.transforms)
    wandb_tracker = instantiate(config.wandb_tracker, OmegaConf.to_container(config))

    trainer = Trainer(
        config=config,
        device=device,
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloaders=dataloaders,
        transforms=transforms,
        wandb_tracker=wandb_tracker,
    )
    trainer.train()


if __name__ == "__main__":
    train()
