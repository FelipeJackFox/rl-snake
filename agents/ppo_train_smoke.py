from __future__ import annotations
import os, argparse, sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

from envs.snake_env import SnakeEnv, SnakeConfig
from utils.progress import TqdmSB3Callback

def make_env(grid_size=12, max_no_food_steps=80, render_mode=None):
    def _thunk():
        env = SnakeEnv(render_mode=render_mode, config=SnakeConfig(grid_size=grid_size, max_no_food_steps=max_no_food_steps))
        return Monitor(env)
    return _thunk

def main():
    print("[SMOKE] starting training…", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--tb", type=str, default="runs")
    args = parser.parse_args()

    os.makedirs(args.tb, exist_ok=True)

    vec_env = DummyVecEnv([make_env() for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024,
        batch_size=256,
        n_epochs=3,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        tensorboard_log=args.tb,
        verbose=1,
    )
    model.set_logger(configure(args.tb, ["stdout", "tensorboard"]))

    tqdm_cb = TqdmSB3Callback()
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=[tqdm_cb])
    print("[SMOKE] done.", flush=True)

if __name__ == "__main__":
    main()
