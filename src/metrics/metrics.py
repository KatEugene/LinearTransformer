import torch
from sacrebleu import corpus_bleu
from torch import nn
from src.utils.globals import PAD_ID


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, label_smoothing):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=PAD_ID)
        self.name = "loss"

    def forward(self, logits, trg_text, **batch):
        return {"loss": self.loss(logits, trg_text)}


class BaseMetric:
    def __init__(self, name, is_global, **config):
        self.name = name
        self.is_global = is_global


class GradNormMetric(BaseMetric):
    def __call__(self, parameters, norm_type=2, **batch):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()


class BLEUMetric(BaseMetric):
    def __init__(self, name, is_global, **config):
        super().__init__(name, is_global, **config)
        self.translations = []
        self.references = []

    def __call__(self, decoded_trg_text, decoded_translation_text, **batch):
        self.references.append(decoded_trg_text)
        self.translations.extend(decoded_translation_text)
        return 0

    def get_score(self):
        bleu_score = corpus_bleu(self.translations, self.references)
        return bleu_score.score
