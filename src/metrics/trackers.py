from datetime import datetime
import numpy as np
import pandas as pd
import wandb


class MetricTracker:
    def __init__(self, *keys, wandb_tracker=None):
        self.wandb_tracker = wandb_tracker
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.wandb_tracker is not None:
            self.wandb_tracker.log_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        if self._data.loc[key, "counts"] != 0:
            self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


class WandbTracker:
    def __init__(self, project_config, project_name, run_id, run_name=None, mode="online", **kwargs):
        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

        wandb.init(
            project=project_name,
            config=project_config,
            name=run_name,
            resume="allow",
            id=run_id,
            mode=mode,
        )

    def set_step(self, step, mode="train"):
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.log_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        return f"{object_name}_{self.mode}"

    def log_scalar(self, scalar_name, scalar):
        wandb.log({self._object_name(scalar_name): scalar}, step=self.step)

    def log_translation(self, src_texts, trg_texts, translation_texts):
        table = wandb.Table(columns=["Source", "Target", "Translation"])
        for src, trg, translation in zip(src_texts, trg_texts, translation_texts):
            table.add_data(src, trg, translation)
        wandb.log({"Translation Results": table}, step=self.step)

    def log_config(self, path):
        artifact = wandb.Artifact(name="config", type="config")
        artifact.add_file(str(path))
        wandb.log_artifact(artifact)
