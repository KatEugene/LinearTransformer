import torch
from tqdm.auto import tqdm

from src.utils.globals import ROOT_PATH


class Inferencer:
    def __init__(self, config, device, model, test_dataloder, transforms):
        self.config = config
        self.device = device
        self.model = model
        self.test_dataloder = test_dataloder
        save_dir = ROOT_PATH / config.inferencer.save_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_path = save_dir / "inference_result"
        self.transforms = transforms

        pretrained_path = ROOT_PATH / config.inferencer.pretrained_path
        self._from_pretrained(pretrained_path)

    def inference(self):
        self.model.eval()
        save_file = self.save_path.open(mode='w')
        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.test_dataloder), desc="test", total=len(self.test_dataloder)):
                self.process_batch(batch)
                for line in batch['generated_texts']:
                    save_file.write(line)
                    save_file.write('\n')

    def process_batch(self, batch):
        self._move_batch_to_device(batch)
        self._transform_batch(batch)
        generated_texts = self.model.inference(**batch)
        batch.update(generated_texts)

    def _from_pretrained(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, self.device)
        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _move_batch_to_device(self, batch):
        for tensor in self.config.trainer.to_device:
            batch[tensor] = batch[tensor].to(self.device)

    def _transform_batch(self, batch):
        for transform_name, transform in self.transforms.items():
            batch[transform_name] = transform(batch[transform_name])
