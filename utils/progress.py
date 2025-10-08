from __future__ import annotations
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import sys

class TqdmSB3Callback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pbar: Optional[tqdm] = None
        self._last_num_timesteps = 0
        self._total_target = None

    def _on_training_start(self) -> None:
        self._total_target = getattr(self.model, "_total_timesteps", None)
        self.pbar = tqdm(
            total=self._total_target,
            desc="PPO training",
            unit="ts",
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
        )
        return None

    def _on_step(self) -> bool:
        if self.pbar is None:
            return True
        delta = self.num_timesteps - self._last_num_timesteps
        if delta > 0:
            self.pbar.update(delta)
            self._last_num_timesteps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            remaining = (self._total_target or self.num_timesteps) - self._last_num_timesteps
            if remaining > 0:
                self.pbar.update(remaining)
            self.pbar.close()
