import torch
import hydra
from hydra.utils import instantiate

from src.utils.helpers import set_random_seed
from src.utils.preprocessing import get_dataloaders
from src.trainer.inferencer import Inferencer


@hydra.main(version_base=None, config_path=".", config_name="config")
def inference(config):
    set_random_seed(config.globals.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders, tokenizer = get_dataloaders(config)
    test_dataloader = dataloaders["valid"]
    model = instantiate(config.model, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer).to(device)
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
    inference()
