from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np


ACTION_NAMES = ("Юг", "Север", "Восток", "Запад", "Посадка", "Высадка")
TAXI_ENV_ID = "Taxi-v4" if "Taxi-v4" in gym.envs.registry else "Taxi-v3"


class TaxiModel:
    def __init__(self) -> None:
        self.env = gym.make(TAXI_ENV_ID)
        self.transitions = self.env.unwrapped.P
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)

    def step(
        self, state: int, action: int, rng: np.random.Generator
    ) -> tuple[int, float, bool]:
        outcomes = self.transitions[state][action]
        probabilities = np.asarray([item[0] for item in outcomes], dtype=float)
        index = int(rng.choice(len(outcomes), p=probabilities / probabilities.sum()))
        _, next_state, reward, terminal = outcomes[index]
        return int(next_state), float(reward), bool(terminal)


def _random_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    indices = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(indices))


def mcts_action(
    model: TaxiModel,
    root_state: int,
    q_values: np.ndarray,
    simulations: int,
    depth: int,
    gamma: float,
    rng: np.random.Generator,
    exploration: float = 1.4,
) -> int:
    """UCT search with random rollouts and Q-value bootstrapping at the leaf."""
    visits: dict[tuple[int, int], np.ndarray] = {}
    values: dict[tuple[int, int], np.ndarray] = {}

    for _ in range(simulations):
        state = root_state
        trajectory: list[tuple[tuple[int, int], int, float]] = []
        terminal = False
        expanded = False

        for level in range(depth):
            key = (state, level)
            action_visits = visits.setdefault(key, np.zeros(model.n_actions, dtype=np.int64))
            action_values = values.setdefault(key, np.zeros(model.n_actions, dtype=float))
            untried = np.flatnonzero(action_visits == 0)
            if untried.size:
                action = int(rng.choice(untried))
                expanded = True
            else:
                total = max(1, int(action_visits.sum()))
                ucb = action_values + exploration * np.sqrt(
                    np.log(total + 1) / np.maximum(action_visits, 1)
                )
                action = _random_argmax(ucb, rng)

            next_state, reward, terminal = model.step(state, action, rng)
            trajectory.append((key, action, reward))
            state = next_state
            if terminal:
                break
            if expanded:
                for rollout_level in range(level + 1, depth):
                    rollout_action = int(rng.integers(model.n_actions))
                    next_state, rollout_reward, terminal = model.step(
                        state, rollout_action, rng
                    )
                    trajectory.append(((state, rollout_level), rollout_action, rollout_reward))
                    state = next_state
                    if terminal:
                        break
                break

        return_value = 0.0 if terminal else float(np.max(q_values[state]))
        for key, action, reward in reversed(trajectory):
            return_value = reward + gamma * return_value
            node_visits = visits.setdefault(key, np.zeros(model.n_actions, dtype=np.int64))
            node_values = values.setdefault(key, np.zeros(model.n_actions, dtype=float))
            node_visits[action] += 1
            node_values[action] += (return_value - node_values[action]) / node_visits[action]

    root_visits = visits.get((root_state, 0), np.zeros(model.n_actions, dtype=np.int64))
    if not np.any(root_visits):
        return _random_argmax(q_values[root_state], rng)
    return _random_argmax(root_visits.astype(float), rng)


@dataclass
class TaxiTrainingResult:
    rewards: np.ndarray
    steps: np.ndarray
    q_values: np.ndarray
    duration: float
    evaluation_reward: float
    success_rate: float


def evaluate_taxi_policy(
    q_values: np.ndarray, episodes: int = 100, seed: int = 10_002
) -> tuple[float, float]:
    env = gym.make(TAXI_ENV_ID)
    rewards: list[float] = []
    successes = 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        for _ in range(200):
            action = int(np.argmax(q_values[int(state)]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                successes += int(terminated and reward == 20)
                break
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), successes / episodes


def train_taxi(
    method: str,
    episodes: int = 300,
    seed: int = 2,
    simulations: int = 25,
    depth: int = 15,
    alpha: float = 0.25,
    gamma: float = 0.95,
    epsilon: float = 0.12,
    evaluation_episodes: int = 100,
) -> TaxiTrainingResult:
    if method not in {"q_learning", "mcts"}:
        raise ValueError(f"Неизвестный метод: {method}")
    env = gym.make(TAXI_ENV_ID)
    model = TaxiModel()
    rng = np.random.default_rng(seed)
    q_values = np.zeros((model.n_states, model.n_actions), dtype=float)
    rewards = np.zeros(episodes, dtype=float)
    steps = np.zeros(episodes, dtype=int)
    started = time.perf_counter()

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        for step_index in range(200):
            state = int(state)
            if rng.random() < epsilon:
                action = int(rng.integers(model.n_actions))
            elif method == "mcts":
                action = mcts_action(
                    model, state, q_values, simulations, depth, gamma, rng
                )
            else:
                action = _random_argmax(q_values[state], rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)
            target = float(reward)
            if not (terminated or truncated):
                target += gamma * np.max(q_values[next_state])
            q_values[state, action] += alpha * (target - q_values[state, action])
            state = next_state
            total += float(reward)
            if terminated or truncated:
                break
        rewards[episode] = total
        steps[episode] = step_index + 1

    duration = time.perf_counter() - started
    evaluation_reward, success_rate = evaluate_taxi_policy(
        q_values, episodes=evaluation_episodes, seed=seed + 20_000
    )
    env.close()
    model.env.close()
    return TaxiTrainingResult(
        rewards=rewards,
        steps=steps,
        q_values=q_values,
        duration=duration,
        evaluation_reward=evaluation_reward,
        success_rate=success_rate,
    )


def mcts_parameter_study(
    depths: Iterable[int],
    simulations: Iterable[int],
    episodes: int = 160,
    seed: int = 2,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for depth in depths:
        for count in simulations:
            result = train_taxi(
                "mcts",
                episodes=episodes,
                seed=seed + depth * 101 + count,
                simulations=count,
                depth=depth,
            )
            rows.append(
                {
                    "Глубина": depth,
                    "Симуляции": count,
                    "Время, с": result.duration,
                    "Средняя награда": result.evaluation_reward,
                    "Успех, %": 100 * result.success_rate,
                }
            )
    return rows


def taxi_policy_rows(q_values: np.ndarray, limit: int = 12) -> list[dict[str, int | str]]:
    env = gym.make(TAXI_ENV_ID)
    rows: list[dict[str, int | str]] = []
    states = np.linspace(0, q_values.shape[0] - 1, limit, dtype=int)
    for state in states:
        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(int(state))
        action = int(np.argmax(q_values[state]))
        rows.append(
            {
                "Состояние": int(state),
                "Такси": f"({taxi_row}, {taxi_col})",
                "Пассажир": int(passenger),
                "Цель": int(destination),
                "Действие": ACTION_NAMES[action],
            }
        )
    env.close()
    return rows


TRUE_RANDOM_WALK_VALUES = np.linspace(-0.9, 0.9, 19)


def _random_walk_episode(rng: np.random.Generator) -> tuple[list[int], list[float]]:
    state = 10
    states = [state]
    rewards = [0.0]
    while state not in {0, 20}:
        state += -1 if rng.random() < 0.5 else 1
        reward = -1.0 if state == 0 else (1.0 if state == 20 else 0.0)
        states.append(state)
        rewards.append(reward)
    return states, rewards


def _backward_td_lambda(
    values: np.ndarray,
    states: list[int],
    rewards: list[float],
    alpha: float,
    lam: float,
) -> None:
    eligibility = np.zeros_like(values)
    for index in range(len(states) - 1):
        state, next_state = states[index], states[index + 1]
        delta = rewards[index + 1] + values[next_state] - values[state]
        eligibility *= lam
        eligibility[state] += 1.0
        values += alpha * delta * eligibility
        values[0] = values[20] = 0.0


def _forward_td_lambda(
    values: np.ndarray,
    states: list[int],
    rewards: list[float],
    alpha: float,
    lam: float,
) -> None:
    snapshot = values.copy()
    changes = np.zeros_like(values)
    terminal_index = len(states) - 1
    lambda_returns = np.zeros(terminal_index, dtype=float)
    lambda_return = 0.0
    for time_index in range(terminal_index - 1, -1, -1):
        next_state = states[time_index + 1]
        lambda_return = rewards[time_index + 1] + (
            (1.0 - lam) * snapshot[next_state] + lam * lambda_return
        )
        lambda_returns[time_index] = lambda_return
    for time_index in range(terminal_index):
        state = states[time_index]
        changes[state] += alpha * (lambda_returns[time_index] - snapshot[state])
    values += changes
    values[0] = values[20] = 0.0


def td_lambda_errors(
    method: str,
    lambdas: Iterable[float],
    alphas: Iterable[float],
    runs: int = 20,
    episodes: int = 10,
    seed: int = 2,
) -> dict[float, np.ndarray]:
    updater = _forward_td_lambda if method == "forward" else _backward_td_lambda
    result: dict[float, np.ndarray] = {}
    for lam in lambdas:
        alpha_errors: list[float] = []
        for alpha in alphas:
            errors: list[float] = []
            for run in range(runs):
                rng = np.random.default_rng(seed + run * 997)
                values = np.zeros(21, dtype=float)
                for episode_index in range(episodes):
                    states, rewards = _random_walk_episode(rng)
                    with np.errstate(over="ignore", invalid="ignore"):
                        updater(values, states, rewards, float(alpha), float(lam))
                    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1e6:
                        errors.extend([5.0] * (episodes - episode_index))
                        break
                    rms_error = float(
                        np.sqrt(np.mean((values[1:20] - TRUE_RANDOM_WALK_VALUES) ** 2))
                    )
                    if rms_error > 5.0:
                        errors.extend([5.0] * (episodes - episode_index))
                        break
                    errors.append(rms_error)
            alpha_errors.append(float(np.mean(errors)))
        result[float(lam)] = np.asarray(alpha_errors)
    return result
