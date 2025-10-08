from __future__ import annotations
import os
import json
import random
from dataclasses import dataclass

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from envs.snake_env import SnakeEnv, SnakeConfig

@dataclass
class PBTConfig:
    population: int = 8
    gen_steps: int = 300_000
    generations: int = 5
    exploit_top_k: int = 2
    eval_episodes: int = 10
    save_dir: str = "models/pbt"

BASE_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "net_arch": [256, 256],
}

MUTATION_FACTORS = {
    "learning_rate": [0.7, 1.0, 1.4],
    "ent_coef": [0.8, 1.0, 1.25],
    "clip_range": [0.9, 1.0, 1.1],
}

def mutate(params: dict) -> dict:
    p = params.copy()
    for k, factors in MUTATION_FACTORS.items():
        p[k] = float(p[k]) * random.choice(factors)
    return p

def train_agent(agent_dir: str, params: dict, steps: int) -> float:
    env = Monitor(SnakeEnv(render_mode=None, config=SnakeConfig()))
    model = PPO(
        "MlpPolicy",
        env,
        **{k: v for k, v in params.items() if k in [
            "learning_rate", "gamma", "gae_lambda", "clip_range", "ent_coef",
            "vf_coef", "n_steps", "batch_size", "n_epochs"
        ]},
        policy_kwargs=dict(net_arch=params["net_arch"]),
        verbose=0,
    )
    model.learn(total_timesteps=steps)
    mean_rew, _ = evaluate_policy(model, env, n_eval_episodes=10)
    os.makedirs(agent_dir, exist_ok=True)
    model.save(os.path.join(agent_dir, "model"))
    with open(os.path.join(agent_dir, "params.json"), "w") as f:
        json.dump(params, f)
    env.close()
    return float(mean_rew)

def run_pbt(cfg: PBTConfig = PBTConfig()):
    os.makedirs(cfg.save_dir, exist_ok=True)
    pop = []
    for i in range(cfg.population):
        params = mutate(BASE_PARAMS) if i > 0 else BASE_PARAMS.copy()
        pop.append({"id": i, "params": params, "fitness": -1e9})

    for g in range(cfg.generations):
        print(f"\n=== Generation {g+1}/{cfg.generations} ===")
        for a in pop:
            agent_dir = os.path.join(cfg.save_dir, f"agent_{a['id']}")
            a["fitness"] = train_agent(agent_dir, a["params"], cfg.gen_steps)
            print(f"Agent {a['id']} fitness: {a['fitness']:.2f}")

        pop.sort(key=lambda x: x["fitness"], reverse=True)
        print("Top fitness:", [round(x["fitness"], 2) for x in pop[:cfg.exploit_top_k]])
        for loser in pop[cfg.exploit_top_k:]:
            winner = random.choice(pop[:cfg.exploit_top_k])
            loser["params"] = mutate(winner["params"])

        with open(os.path.join(cfg.save_dir, f"gen_{g+1}.json"), "w") as f:
            json.dump(pop, f, indent=2)

    best = max(pop, key=lambda x: x["fitness"])
    print("Best agent:", best["id"], "fitness", best["fitness"])

if __name__ == "__main__":
    run_pbt()
