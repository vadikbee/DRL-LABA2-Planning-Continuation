import numpy as np
import torch

from algorithms import (
    TAXI_ENV_ID,
    TaxiModel,
    mcts_action,
    td_lambda_errors,
    train_taxi,
)
from nfq import NFQNetwork, train_nfq


def test_taxi_model_and_mcts_return_valid_action():
    model = TaxiModel()
    q_values = np.zeros((model.n_states, model.n_actions))
    action = mcts_action(
        model,
        root_state=42,
        q_values=q_values,
        simulations=3,
        depth=4,
        gamma=0.95,
        rng=np.random.default_rng(2),
    )
    model.env.close()
    assert TAXI_ENV_ID in {"Taxi-v3", "Taxi-v4"}
    assert 0 <= action < 6


def test_td_lambda_grid_shape_and_finiteness():
    result = td_lambda_errors(
        "backward", (0.0, 0.8, 1.0), (0.1, 0.2), runs=2, episodes=2
    )
    assert set(result) == {0.0, 0.8, 1.0}
    assert all(values.shape == (2,) for values in result.values())
    assert all(np.isfinite(values).all() for values in result.values())


def test_taxi_training_and_nfq_smoke():
    tabular = train_taxi("q_learning", episodes=2, evaluation_episodes=2)
    nfq = train_nfq(
        epochs=1,
        episodes_per_epoch=1,
        batch_size=8,
        evaluation_episodes=2,
    )
    assert tabular.rewards.shape == (2,)
    assert nfq.rewards.shape == (1,)
    assert NFQNetwork()(torch.tensor([0, 499])).shape == (2, 6)
