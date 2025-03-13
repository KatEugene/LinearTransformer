import torch
from numpy import inf
from tqdm.auto import tqdm
from functools import partial

from src.utils.globals import ROOT_PATH
from src.metrics import MetricTracker


class Trainer:
    def __init__(self, model, criterion, metrics, optimizer, lr_scheduler, config, device, dataloaders, transforms, wandb_tracker):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.train_dataloader = dataloaders["train"]
        self.valid_dataloader = dataloaders["valid"]
        self.transforms = transforms
        self.wandb_tracker = wandb_tracker
        self.train_tracker = MetricTracker(
            self.criterion.name,
            *[metric.name for metric in self.metrics.train],
            wandb_tracker=self.wandb_tracker
        )
        self.valid_tracker = MetricTracker(
            self.criterion.name,
            *[metric.name for metric in self.metrics.inference],
            wandb_tracker=self.wandb_tracker
        )

        self.mode = 'train'
        self.epochs = self.config.trainer.n_epochs
        self.epoch_period = self.config.trainer.epoch_period
        self.start_epoch = 1
        self.last_epoch = 0
        self.epoch_len = len(self.train_dataloader)
        self.log_period = config.trainer.log_period
        self.save_period = self.config.trainer.save_period

        self.best_metric = inf
        self.best_metric_name = 'loss'
        self.early_stop = self.config.trainer.early_stop
        if self.early_stop is None:
            self.early_stop = inf

        self.checkpoint_dir = ROOT_PATH / config.trainer.checkpoint_dir / config.wandb_tracker.run_name
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        if config.trainer.resume_from is not None:
            self._resume_checkpoint(self.checkpoint_dir / config.trainer.resume_from)

    def train(self):
        try:
            self._train()
        except Exception:
            self._save_checkpoint(self.last_epoch)
            raise

    def _train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            logs = self._train_epoch(epoch)
            not_improved_count = self._check_perfomance(logs, not_improved_count)
            is_best = (not_improved_count == 0)
            if epoch % self.save_period == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
            if not_improved_count >= self.early_stop:
                break

    def _train_epoch(self, epoch):
        self.last_epoch = epoch
        self.mode = 'train'
        self.model.train()
        self.train_tracker.reset()
        self.wandb_tracker.set_step((epoch - 1) * self.epoch_len)
        self.wandb_tracker.log_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=len(self.train_dataloader))):
            self.process_batch(batch, self.train_tracker)

            if batch_idx % self.log_period == 0:
                self.wandb_tracker.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.wandb_tracker.log_scalar("learning_rate", self.lr_scheduler.get_last_lr()[0])
                self._log_batch(batch, self.train_tracker, epoch)
                logs = self.train_tracker.result()
                self.train_tracker.reset()

        val_logs = self._valid_epoch(epoch)
        logs.update(**{f"valid_{name}": value for name, value in val_logs.items()})
        return logs

    def _valid_epoch(self, epoch):
        self.mode = 'inference'
        self.model.eval()
        self.valid_tracker.reset()
        logging_global = (epoch % self.epoch_period == 0)

        with torch.no_grad():
            for _, batch in tqdm(enumerate(self.valid_dataloader), desc="valid", total=len(self.valid_dataloader)):
                self.process_batch(batch, self.valid_tracker, logging_global)

            if logging_global:
                for metric in self.metrics[self.mode]:
                    if metric.is_global:
                        self.valid_tracker.update(metric.name, metric.get_score())

            self.wandb_tracker.set_step(epoch * self.epoch_len, "valid")
            self._log_batch(batch, self.valid_tracker, epoch, logging_global)

        return self.valid_tracker.result()

    def _check_perfomance(self, logs, not_improved_count):
        if logs[self.best_metric_name] <= self.best_metric:
            self.best_metric = logs[self.best_metric_name]
            not_improved_count = 0
        else:
            not_improved_count += 1
        return not_improved_count

    def _log_batch(self, batch, metric_tracker, epoch, logging_global=False):
        banned_names = self._get_banned_metrics(epoch)
        for metric_name in metric_tracker.keys():
            if metric_name not in banned_names:
                self.wandb_tracker.log_scalar(metric_name, metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            "model": type(self.model).__name__,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        if is_best:
            path = self.checkpoint_dir / "model_best.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth"
        torch.save(state, path)

    def _resume_checkpoint(self, resume_path):
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def process_batch(self, batch, metric_tracker, logging_global=False):
        self._move_batch_to_device(batch)
        self._transform_batch(batch)

        if self.mode == 'train':
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        batch['logits'] = batch['logits'][:, :-1].transpose(1, 2)
        batch['output_ids'] = batch['input_ids'][:, 1:]
        loss = self.criterion(**batch)
        batch.update(loss)

        if self.mode == 'train':
            batch['loss'].backward()
            self.optimizer.step()
            self.lr_scheduler.step()

        if logging_global:
            decoded_texts = self.model.inference(**batch)
            batch.update(decoded_texts)

        metric_tracker.update(self.criterion.name, batch['loss'].item())

        batch['parameters'] = self.model.parameters()
        for metric in self.metrics[self.mode]:
            if not metric.is_global or logging_global:
                n = not metric.is_global
                metric_tracker.update(metric.name, metric(**batch), n)

    def _move_batch_to_device(self, batch):
        for tensor in self.config.trainer.to_device:
            batch[tensor] = batch[tensor].to(self.device)

    def _transform_batch(self, batch):
        for transform_name, transform in self.transforms[self.mode].items():
            batch[transform_name] = transform(batch[transform_name])

    def _is_metric_banned(self, metric, epoch):
        not_global = not metric.is_global
        correct_epoch = (metric.is_global and epoch % self.epoch_period == 0)
        return not (not_global or correct_epoch)

    def _get_banned_metrics(self, epoch):
        filter_banned_metrics = partial(self._is_metric_banned, epoch=epoch)
        banned_metrics = list(filter(filter_banned_metrics, self.metrics[self.mode]))
        banned_names = list(map(lambda metric: metric.name, banned_metrics))
        return banned_names
