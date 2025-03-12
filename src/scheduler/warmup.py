from torch.optim.lr_scheduler import _LRScheduler


class WarmUpScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, epoch_len, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps * epoch_len
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lr = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        return [lr for _ in self.optimizer.param_groups]
