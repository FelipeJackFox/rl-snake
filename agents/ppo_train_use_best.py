from __future__ import annotations
import os, json, ast, argparse, sys
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.logger import configure

from envs.snake_env import SnakeEnv, SnakeConfig
from utils.callbacks import make_eval_callback
from utils.progress import TqdmSB3Callback

DEFAULT_BEST = {
    # fallback to your printed Optuna result if file missing/unreadable
    "learning_rate": 4.328508741395367e-04,
    "gamma": 0.9927348701326604,
    "gae_lambda": 0.9237595317731032,
    "clip_range": 0.16491988479393566,
    "ent_coef": 0.003013662487593483,
    "vf_coef": 0.49612561369183333,
    "n_steps": 1536,
    "batch_size": 128,
    "n_epochs": 5,
    "net_arch": [128, 128],
}

def make_env(grid_size=12, max_no_food_steps=80, render_mode=None):
    def _thunk():
        env = SnakeEnv(render_mode=render_mode, config=SnakeConfig(grid_size=grid_size, max_no_food_steps=max_no_food_steps))
        return Monitor(env)
    return _thunk

def load_best_params(path: str) -> dict:
    print(f"[INFO] reading params from: {path}", flush=True)
    if not os.path.exists(path):
        print("[WARN] params file not found; using DEFAULT_BEST.", flush=True)
        return DEFAULT_BEST.copy()
    raw = open(path, "r", encoding="utf-8").read().strip()
    if not raw:
        print("[WARN] params file empty; using DEFAULT_BEST.", flush=True)
        return DEFAULT_BEST.copy()
    try:
        params = json.loads(raw)
        print("[INFO] parsed JSON params.", flush=True)
    except json.JSONDecodeError:
        params = ast.literal_eval(raw)
        print("[INFO] parsed literal params.", flush=True)
    if "net_arch" in params and not isinstance(params["net_arch"], list):
        params["net_arch"] = list(params["net_arch"])
    return params

def main():
    print("[RUN] ppo_train_use_best starting…", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default="models/best_params.txt")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--tb", type=str, default="runs")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--max_no_food_steps", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.tb, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    best = load_best_params(args.params)
    print("[INFO] best params:", best, flush=True)

    vec_env = DummyVecEnv([make_env(args.grid_size, args.max_no_food_steps) for _ in range(args.n_envs)])
    vec_env = VecMonitor(vec_env)
    eval_env = DummyVecEnv([make_env(args.grid_size, args.max_no_food_steps)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(best["learning_rate"]),
        gamma=float(best["gamma"]),
        gae_lambda=float(best["gae_lambda"]),
        clip_range=float(best["clip_range"]),
        ent_coef=float(best["ent_coef"]),
        vf_coef=float(best["vf_coef"]),
        n_steps=int(best["n_steps"]),
        batch_size=int(best["batch_size"]),
        n_epochs=int(best["n_epochs"]),
        policy_kwargs=dict(net_arch=best["net_arch"]),
        max_grad_norm=0.5,
        tensorboard_log=args.tb,
        verbose=1,
    )
    model.set_logger(configure(args.tb, ["stdout", "tensorboard"]))
    eval_cb = make_eval_callback(eval_env, save_path=args.models)
    tqdm_cb = TqdmSB3Callback()

    print(f"[RUN] training for {args.timesteps} timesteps on {args.n_envs} env(s)…", flush=True)
    model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=[eval_cb, tqdm_cb])
    print("[RUN] training finished, saving final_model…", flush=True)

    model.save(os.path.join(args.models, "final_model"))
    vec_env.close(); eval_env.close()
    print("[RUN] done.", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("[ERROR] crashed with exception:", e, file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
