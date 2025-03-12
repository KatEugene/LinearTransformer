import torch
from sacrebleu import corpus_bleu
from torch import nn


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, label_smoothing, ignore_index):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)
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
