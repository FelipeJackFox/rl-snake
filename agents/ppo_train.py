from __future__ import annotations
import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

from envs.snake_env import SnakeEnv, SnakeConfig
from utils.callbacks import make_eval_callback
from utils.progress import TqdmSB3Callback  # tqdm progress bar


def make_env(render_mode=None):
    def _thunk():
        env = SnakeEnv(render_mode=render_mode, config=SnakeConfig())
        return Monitor(env)
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--tb", type=str, default="runs")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--n_envs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.tb, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    vec_env = DummyVecEnv([make_env(None) for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)
    eval_env = DummyVecEnv([make_env(None)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=args.tb,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # Logger & callbacks
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.set_logger(configure(args.tb, ["stdout", "tensorboard"]))
    eval_cb = make_eval_callback(eval_env, save_path=args.models)
    tqdm_cb = TqdmSB3Callback()

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,           # SB3’s built-in bar
        callback=[eval_cb, tqdm_cb], # our tqdm bar + eval checkpointing
    )

    model.save(os.path.join(args.models, "final_model"))

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
