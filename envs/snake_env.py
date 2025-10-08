from __future__ import annotations
import time
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


@dataclass
class SnakeConfig:
    grid_size: int = 15
    render_scale: int = 32
    max_no_food_steps: int = 150
    step_penalty: float = -0.01
    eat_reward: float = 1.0
    death_penalty: float = -1.0
    progress_bonus: float = 0.1


REL_ACTIONS = {0: -1, 1: 0, 2: 1}          # 0=left,1=straight,2=right
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 0=up,1=right,2=down,3=left


class SnakeEnv(gym.Env):
    # bump default FPS a bit for responsiveness
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, config: Optional[SnakeConfig] = None):
        super().__init__()
        self.cfg = config or SnakeConfig()
        self.render_mode = render_mode
        self.grid = self.cfg.grid_size

        # 4 dir onehot + 2 rel food + 3 dangers + 4 wall dists + len + steps = 15
        self.obs_size = 15
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.window = None
        self.clock = None
        self.font = None

        self.reset(seed=None)

    # ---------- helpers ----------
    def _spawn_food(self):
        empty = set((x, y) for x in range(self.grid) for y in range(self.grid)) - set(self.snake)
        self.food = random.choice(list(empty))

    def _reset_snake(self):
        cx = self.grid // 2
        cy = self.grid // 2
        self.snake: List[Tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.dir = 1  # right

    def _danger_in_dir(self, dir_idx: int) -> bool:
        dx, dy = DIRS[dir_idx]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy
        if nx < 0 or ny < 0 or nx >= self.grid or ny >= self.grid:
            return True
        if (nx, ny) in self.snake[:-1]:
            return True
        return False

    def _relative_dangers(self):
        ahead = self._danger_in_dir(self.dir)
        left  = self._danger_in_dir((self.dir - 1) % 4)
        right = self._danger_in_dir((self.dir + 1) % 4)
        return ahead, left, right

    def _wall_distances(self):
        hx, hy = self.snake[0]
        up    = hy / (self.grid - 1)
        right = (self.grid - 1 - hx) / (self.grid - 1)
        down  = (self.grid - 1 - hy) / (self.grid - 1)
        left  = hx / (self.grid - 1)
        return np.array([up, right, down, left], dtype=np.float32)

    def _obs(self):
        dir_onehot = np.zeros(4, dtype=np.float32); dir_onehot[self.dir] = 1.0
        hx, hy = self.snake[0]; fx, fy = self.food
        rel = np.array([(fx - hx) / (self.grid - 1), (fy - hy) / (self.grid - 1)], dtype=np.float32)
        ahead, left, right = self._relative_dangers()
        dangers = np.array([ahead, left, right], dtype=np.float32)
        wall = self._wall_distances()
        length_norm = np.array([len(self.snake) / (self.grid * 0.6)], dtype=np.float32)
        steps_norm  = np.array([self.steps_since_food / max(1, self.cfg.max_no_food_steps)], dtype=np.float32)
        return np.concatenate([dir_onehot, rel, dangers, wall, length_norm, steps_norm]).astype(np.float32)

    # ---------- gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._reset_snake()
        self._spawn_food()
        self.steps_since_food = 0
        self.score = 0
        self.prev_score = 0
        self.total_steps = 0

        if self.render_mode == "human":
            self._init_render()
        return self._obs(), {}

    def step(self, action: int):
        self.dir = (self.dir + REL_ACTIONS[int(action)]) % 4
        dx, dy = DIRS[self.dir]
        hx, hy = self.snake[0]
        nx, ny = hx + dx, hy + dy

        reward = self.cfg.step_penalty
        terminated = False
        truncated = False

        if nx < 0 or ny < 0 or nx >= self.grid or ny >= self.grid:
            reward += self.cfg.death_penalty
            terminated = True
        else:
            new_head = (nx, ny)
            if new_head in self.snake[:-1]:
                reward += self.cfg.death_penalty
                terminated = True
            else:
                self.snake.insert(0, new_head)
                if new_head == self.food:
                    self.score += 1
                    reward += self.cfg.eat_reward
                    if self.cfg.progress_bonus:
                        reward += self.cfg.progress_bonus * (self.score - self.prev_score)
                    self.prev_score = self.score
                    self._spawn_food()
                    self.steps_since_food = 0
                else:
                    self.snake.pop()
                    self.steps_since_food += 1

        if self.steps_since_food >= self.cfg.max_no_food_steps:
            truncated = True

        self.total_steps += 1

        if self.render_mode == "human":
            self.render()

        return self._obs(), float(reward), terminated, truncated, {"score": self.score}

    # ---------- rendering ----------
    def _init_render(self):
        if not pygame.get_init():
            pygame.init()
        size = self.grid * self.cfg.render_scale
        # NOTE: keep window creation on main thread
        self.window = pygame.display.set_mode((size, size))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

    def render(self):
        # Make sure window exists
        if self.window is None:
            self._init_render()

        # 🚑 Always pump & drain events (prevents “Not Responding” on Windows)
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill((30, 30, 30))
        s = self.cfg.render_scale

        # Food
        fx, fy = self.food
        pygame.draw.rect(self.window, (200, 80, 80), (fx * s, fy * s, s, s))

        # Snake
        for i, (x, y) in enumerate(self.snake):
            c = (80, 200, 120) if i == 0 else (60, 160, 100)
            pygame.draw.rect(self.window, c, (x * s, y * s, s, s))

        # HUD
        hud = f"Score:{self.score}  Len:{len(self.snake)}  StepsSinceFood:{self.steps_since_food}"
        surf = self.font.render(hud, True, (230, 230, 230))
        self.window.blit(surf, (6, 6))

        pygame.display.flip()
        # Cap FPS and yield a moment to the OS
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(0.001)

    def close(self):
        # Robust teardown
        try:
            if self.window:
                pygame.display.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
        self.window = None
        self.clock = None
        self.font = None
