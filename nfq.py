from __future__ import annotations

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algorithms import TAXI_ENV_ID


class NFQNetwork(nn.Module):
    def __init__(self, n_states: int = 500, n_actions: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = F.one_hot(states.long(), num_classes=500).float()
        return self.network(encoded)


@dataclass
class NFQResult:
    rewards: np.ndarray
    losses: np.ndarray
    evaluation_reward: float
    success_rate: float
    duration: float


def _evaluate_network(model: nn.Module, episodes: int, seed: int) -> tuple[float, float]:
    env = gym.make(TAXI_ENV_ID)
    totals: list[float] = []
    successes = 0
    model.eval()
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        for _ in range(200):
            with torch.no_grad():
                action = int(model(torch.tensor([int(state)])).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                successes += int(terminated and reward == 20)
                break
        totals.append(total)
    env.close()
    return float(np.mean(totals)), successes / episodes


def train_nfq(
    epochs: int = 20,
    episodes_per_epoch: int = 20,
    epsilon: float = 0.3,
    batch_size: int = 64,
    gamma: float = 0.95,
    learning_rate: float = 0.003,
    hidden: int = 64,
    seed: int = 2,
    evaluation_episodes: int = 80,
) -> NFQResult:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = gym.make(TAXI_ENV_ID)
    model = NFQNetwork(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    memory: list[tuple[int, int, float, int, bool]] = []
    epoch_rewards = np.zeros(epochs, dtype=float)
    epoch_losses = np.zeros(epochs, dtype=float)
    started = time.perf_counter()

    for epoch in range(epochs):
        collected_rewards: list[float] = []
        model.eval()
        for episode in range(episodes_per_epoch):
            state, _ = env.reset(seed=seed + epoch * episodes_per_epoch + episode)
            total = 0.0
            for _ in range(150):
                if rng.random() < epsilon:
                    action = int(rng.integers(6))
                else:
                    with torch.no_grad():
                        action = int(model(torch.tensor([int(state)])).argmax(dim=1).item())
                next_state, reward, terminated, truncated, _ = env.step(action)
                memory.append(
                    (int(state), action, float(reward), int(next_state), bool(terminated or truncated))
                )
                if len(memory) > 15_000:
                    memory.pop(0)
                state = next_state
                total += float(reward)
                if terminated or truncated:
                    break
            collected_rewards.append(total)
        epoch_rewards[epoch] = float(np.mean(collected_rewards))

        losses: list[float] = []
        model.train()
        updates = max(12, min(48, len(memory) // max(batch_size, 1)))
        for _ in range(updates):
            indices = rng.choice(len(memory), size=min(batch_size, len(memory)), replace=False)
            batch = [memory[int(index)] for index in indices]
            states = torch.tensor([item[0] for item in batch])
            actions = torch.tensor([item[1] for item in batch])
            rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
            next_states = torch.tensor([item[3] for item in batch])
            terminals = torch.tensor([item[4] for item in batch], dtype=torch.float32)
            with torch.no_grad():
                targets = rewards + gamma * (1.0 - terminals) * model(next_states).max(dim=1).values
            predictions = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses[epoch] = float(np.mean(losses)) if losses else 0.0

    duration = time.perf_counter() - started
    evaluation_reward, success_rate = _evaluate_network(
        model, evaluation_episodes, seed + 30_000
    )
    env.close()
    return NFQResult(
        rewards=epoch_rewards,
        losses=epoch_losses,
        evaluation_reward=evaluation_reward,
        success_rate=success_rate,
        duration=duration,
    )
