from __future__ import annotations
import os, warnings
from functools import partial
import optuna
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from envs.snake_env import SnakeEnv, SnakeConfig

warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def make_env():
    # smaller grid & fewer no-food steps to speed up
    return Monitor(SnakeEnv(render_mode=None, config=SnakeConfig(grid_size=12, max_no_food_steps=80)))

def objective(trial: optuna.Trial, timesteps: int = 100_000) -> float:
    env = make_env()

    lr = trial.suggest_float("learning_rate", 3e-4, 1e-3, log=True)  # tight ranges for speed
    gamma = trial.suggest_float("gamma", 0.97, 0.995)
    gae = trial.suggest_float("gae_lambda", 0.92, 0.97)
    clip = trial.suggest_float("clip_range", 0.15, 0.25)
    ent = trial.suggest_float("ent_coef", 0.0, 0.01)
    vf = trial.suggest_float("vf_coef", 0.4, 0.8)
    n_steps = trial.suggest_int("n_steps", 1024, 2048, step=512)
    batch = trial.suggest_categorical("batch_size", [128, 256])
    epochs = trial.suggest_int("n_epochs", 3, 5)
    net_t = trial.suggest_categorical("net_arch", [(128, 128), (256, 128)])
    net = list(net_t)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr, gamma=gamma, gae_lambda=gae, clip_range=clip,
        ent_coef=ent, vf_coef=vf, n_steps=n_steps, batch_size=batch, n_epochs=epochs,
        policy_kwargs=dict(net_arch=net),
        verbose=0,
    )

    model.learn(total_timesteps=timesteps, progress_bar=True)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, render=False)  # fewer eval episodes
    env.close()
    return float(mean_reward)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    study = optuna.create_study(direction="maximize")
    n_trials = 3  # only 3 quick trials

    with tqdm(total=n_trials, desc="Optuna trials", unit="trial") as pbar:
        def after_each_trial(study_cb, trial_cb):
            pbar.update(1)
        study.optimize(partial(objective, timesteps=100_000), n_trials=n_trials, callbacks=[after_each_trial])

    print("\nBest trial params:")
    print(study.best_trial.params)
    with open("models/best_params.txt", "w") as f:
        f.write(str(study.best_trial.params))
