import torch
from src.utils.globals import ROOT_PATH
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.trainer.trainer import Trainer
from src.trainer.inferencer import Inferencer
from src.utils.helpers import set_random_seed
from src.utils.preprocessing import get_dataloaders


@hydra.main(version_base=None, config_path=".", config_name="config")
def train(config):
    set_random_seed(config.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders, decoders = get_dataloaders(config)
    model = instantiate(config.model, decoders=decoders).to(device)
    criterion = instantiate(config.loss_function)
    metrics = instantiate(config.metrics)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer, epoch_len=len(dataloaders["train"]))
    transforms = instantiate(config.transforms)
    wandb_tracker = instantiate(config.wandb_tracker, OmegaConf.to_container(config))
    wandb_tracker.log_config(ROOT_PATH / "config.yaml")
    wandb_tracker.log_config(ROOT_PATH / "src/model/transformer.py")
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


@hydra.main(version_base=None, config_path=".", config_name="config")
def inference(config):
    set_random_seed(config.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders, decoders = get_dataloaders(config)
    test_dataloader = dataloaders["test"]
    model = instantiate(config.model, decoders=decoders).to(device)
    transforms = instantiate(config.transforms.inference)

    inferencer = Inferencer(
        config=config,
        device=device,
        model=model,
        test_dataloder=test_dataloader,
        transforms=transforms,
    )
    inferencer.inference()


if __name__ == "__main__":
    train()
    inference()
