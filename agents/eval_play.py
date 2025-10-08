from __future__ import annotations
import os
import argparse
import time
import pygame

from stable_baselines3 import PPO
from envs.snake_env import SnakeEnv, SnakeConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/best_model.zip")
    parser.add_argument("--fps", type=int, default=20, help="Target FPS cap for eval")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        alt = "models/final_model.zip"
        if os.path.exists(alt):
            args.model = alt
        else:
            raise FileNotFoundError("No model found at models/best_model.zip or models/final_model.zip")

    env = SnakeEnv(render_mode="human", config=SnakeConfig())
    # keep a local clock here too, in case render didn't tick (safety)
    local_clock = pygame.time.Clock()

    model = PPO.load(args.model)
    obs, _ = env.reset()

    try:
        while True:
            if env.window is None:
                break

            # Pump events proactively in the main loop too
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(int(action))
            if done or trunc:
                obs, _ = env.reset()

            # Secondary FPS cap/yield (helps when render is temporarily skipped)
            local_clock.tick(args.fps)
            time.sleep(0.001)
    finally:
        env.close()


if __name__ == "__main__":
    main()
