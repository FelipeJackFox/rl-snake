from __future__ import annotations
import os
from stable_baselines3.common.callbacks import EvalCallback

def make_eval_callback(eval_env, save_path: str = "models", best_prefix: str = "best"):
    os.makedirs(save_path, exist_ok=True)
    return EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
