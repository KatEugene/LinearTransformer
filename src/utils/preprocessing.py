import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from src.utils.globals import *
from datasets import load_dataset


class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data, max_seq_len, max_vocab_size, vocabs=None):
        self.max_seq_len = max_seq_len
        self.max_vocab_size = max_vocab_size

        self.vocabs = vocabs
        if self.vocabs is None:
            self.vocabs = dict()

        self.src_data = self._build(src_data, "src")
        self.trg_data = self._build(trg_data, "trg")

    def _build(self, data, data_type):
        if data is None:
            return []
        tokenized_data = [line.split() for line in data]

        if self.vocabs.get(data_type) is None:
            self.vocabs[data_type] = build_vocab_from_iterator(
                iter(tokenized_data),
                specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN],
                max_tokens=self.max_vocab_size,
            )
        return self._preprocess_data(tokenized_data, self.vocabs.get(data_type).get_stoi())

    def _preprocess_data(self, tokenized_data, vocab):
        data = []
        for text in tokenized_data:
            token_ids = [vocab.get(token, UNK_ID) for token in text]
            token_ids = token_ids[:self.max_seq_len - 2]
            token_ids = [SOS_ID] + token_ids + [EOS_ID]
            padded_token_ids = token_ids + [PAD_ID] * (self.max_seq_len - len(token_ids))
            data.append(padded_token_ids)
        return data

    def get_vocabs(self):
        return self.vocabs

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        item = dict()
        item["src_text"] = torch.tensor(self.src_data[idx], dtype=torch.long)
        if self.trg_data:
            item["trg_text"] = torch.tensor(self.trg_data[idx], dtype=torch.long)
        else:
            item["trg_text"] = torch.tensor([])
        return item


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


def collate_fn(dataset_items: list[dict]):
    result_batch = {}
    result_batch["src_text"] = torch.stack([elem["src_text"] for elem in dataset_items])
    result_batch["trg_text"] = torch.stack([elem["trg_text"] for elem in dataset_items])
    return result_batch


def get_dataloaders(config):
    train_dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    valid_dataset = load_dataset("allenai/c4", "realnewslike", split="valid", streaming=True)

    BATCH_SIZE = config.dataloader.batch_size
    MAX_SEQ_LEN = config.dataloader.max_seq_len
    TRAIN_SIZE = config.dataloader.train_size / 100
    MAX_VOCAB_SIZE = config.dataloader.max_vocab_size

    src_train, trg_train = load_data(ROOT_PATH / "data/train.de-en.de"), load_data(ROOT_PATH / "data/train.de-en.en")
    N = len(src_train)
    shuffle_perm = np.random.permutation(N)[:int(N * TRAIN_SIZE)]
    src_train, trg_train = np.array(src_train)[shuffle_perm], np.array(trg_train)[shuffle_perm]

    src_valid, trg_valid = load_data(ROOT_PATH / "data/val.de-en.de"), load_data(ROOT_PATH / "data/val.de-en.en")
    src_test, trg_test = load_data(ROOT_PATH / "data/test1.de-en.de"), None

    train_dataset = TranslationDataset(src_train, trg_train, MAX_SEQ_LEN, MAX_VOCAB_SIZE)
    vocabs = train_dataset.get_vocabs()
    valid_dataset = TranslationDataset(src_valid, trg_valid, MAX_SEQ_LEN, MAX_VOCAB_SIZE, vocabs)
    test_dataset = TranslationDataset(src_test, trg_test, MAX_SEQ_LEN, MAX_VOCAB_SIZE, vocabs)

    dataloaders = {}
    dataloaders["train"] = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
    dataloaders["valid"] = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)
    dataloaders["test"] = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False)

    return dataloaders, vocabs
