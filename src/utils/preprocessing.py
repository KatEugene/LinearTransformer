import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils.globals import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN


class C4Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.input_ids = []
        self.attention_masks = []

        for item in dataset:
            text = f"{SOS_TOKEN} {item['text']} {EOS_TOKEN}"
            encoding = self.tokenizer(
                text,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoding["input_ids"].squeeze(0))
            self.attention_masks.append(encoding["attention_mask"].squeeze(0))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = dict()
        item["input_ids"] = torch.tensor(self.input_ids[idx], dtype=torch.long)
        item["attention_mask"] = torch.tensor(self.attention_masks[idx], dtype=torch.long)
        return item


def collate_fn(dataset_items: list[dict]):
    result_batch = {}
    result_batch["input_ids"] = torch.stack([elem["input_ids"] for elem in dataset_items])
    result_batch["attention_mask"] = torch.stack([elem["attention_mask"] for elem in dataset_items])
    return result_batch


def get_dataloaders(config):
    BATCH_SIZE = config.dataloader.batch_size
    MAX_SEQ_LEN = config.dataloader.max_seq_len
    TRAIN_SIZE = config.dataloader.train_size

    train_dataset = load_dataset("allenai/c4", "st", split="train")
    valid_dataset = load_dataset("allenai/c4", "st", split="validation")

    train_dataset = train_dataset.select(range(TRAIN_SIZE))
    valid_dataset = valid_dataset.select(range(TRAIN_SIZE))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": PAD_TOKEN,
        "bos_token": SOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "unk_token": UNK_TOKEN,
    })

    train_dataset = C4Dataset(train_dataset, tokenizer, MAX_SEQ_LEN)
    valid_dataset = C4Dataset(valid_dataset, tokenizer, MAX_SEQ_LEN)

    dataloaders = {}
    dataloaders["train"] = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    dataloaders["valid"] = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)

    return dataloaders, tokenizer
