import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "method" / "ablation" / "run_wc_two_targets_multiexplore.py"
SPEC = importlib.util.spec_from_file_location("unified_multiexplore", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selected_vae_quality_contract():
    config_path = PROJECT_ROOT / "config" / "selected_vae_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert (PROJECT_ROOT / config["model"]).exists()
    assert config["reference_temperature"] == 1.0
    assert config["quality_20k"]["validity"] >= 0.93
    assert config["quality_20k"]["novelty_unique"] >= 0.93
    assert set(t for _, t, _ in MODULE.DEFAULT_CHANNELS).issubset(
        set(config["allowed_exploration_temperatures"])
    )


def test_adaptive_channel_allocation_uses_hv_utility():
    baseline = MODULE.build_epoch_channels(0, 100, 120, np.zeros(6))
    favored = MODULE.build_epoch_channels(
        0, 100, 120, np.asarray([0, 0, 0, 0, 0, 1.0], dtype=float)
    )
    baseline_counts = Counter(name for name, _, _ in baseline)
    favored_counts = Counter(name for name, _, _ in favored)
    assert len(baseline) == 120
    assert len(favored) == 120
    assert favored_counts["F_strong_explore"] > baseline_counts["F_strong_explore"]


def test_channel_context_is_bounded_and_distinguishable():
    contexts = {
        (round(temp / 1.05, 6), round(step / 1.5, 6))
        for _, temp, step in MODULE.DEFAULT_CHANNELS
    }
    assert len(contexts) >= 4
    assert all(0.0 < temp <= 1.0 and 0.0 < step <= 1.0 for temp, step in contexts)


def test_preference_floor_preserves_both_objectives():
    integrator = MODULE.MultiExploreIntegrator.__new__(MODULE.MultiExploreIntegrator)
    integrator.preference_floor = 0.10
    weights = integrator._normalize_weights(np.asarray([1.0, 0.0]))
    assert np.allclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.10 - 1e-7)
    assert np.allclose(weights, [0.9, 0.1])


def test_trajectory_step_scale_preserves_k1_and_normalizes_longer_paths():
    assert np.isclose(MODULE.trajectory_step_scale(0.08, 1.5, 1, "sqrt"), 0.12)
    assert np.isclose(
        MODULE.trajectory_step_scale(0.08, 1.5, 4, "sqrt"), 0.06
    )
    assert np.isclose(
        MODULE.trajectory_step_scale(0.08, 1.5, 4, "linear"), 0.03
    )


def test_terminal_reward_is_stored_once_and_trajectory_boundaries_are_explicit():
    class RecordingAgent:
        def __init__(self):
            self.records = []

        def store_transition_multi(self, *args, **kwargs):
            self.records.append((args, kwargs))

    agent = RecordingAgent()
    transitions = [
        {
            "policy_state": np.full(4, step, dtype=np.float32),
            "action": np.ones(2, dtype=np.float32),
            "log_prob": -0.1,
            "values": np.zeros(2, dtype=np.float32),
            "entropy": 0.2,
        }
        for step in range(3)
    ]
    MODULE.store_terminal_trajectory(
        agent,
        transitions,
        np.asarray([0.7, 0.8], dtype=np.float32),
        np.asarray([0.5, 0.5], dtype=np.float32),
        auxiliary_reward=0.4,
    )
    assert len(agent.records) == 3
    assert np.allclose(agent.records[0][0][2], [0.0, 0.0])
    assert np.allclose(agent.records[1][0][2], [0.0, 0.0])
    assert np.allclose(agent.records[2][0][2], [0.7, 0.8])
    assert [record[0][5] for record in agent.records] == [False, False, True]
    assert [record[1]["auxiliary_reward"] for record in agent.records] == [0.0, 0.0, 0.4]


def test_multi_critic_gae_propagates_terminal_scores_without_crossing_done():
    agent = MODULE.TrajectoryMultiCriticPPOAgent(
        state_dim=4,
        action_dim=2,
        num_obj=2,
        gamma=0.9,
        gae_lambda=1.0,
        mini_batch_size=1,
        device="cpu",
    )
    rewards = MODULE.torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]
    )
    values = MODULE.torch.zeros_like(rewards)
    dones = MODULE.torch.tensor([0.0, 0.0, 1.0, 1.0])
    advantages = agent._compute_trajectory_gae(rewards, values, dones).numpy()
    assert np.allclose(advantages[0], [0.81, 1.62], atol=1e-6)
    assert np.allclose(advantages[1], [0.9, 1.8], atol=1e-6)
    assert np.allclose(advantages[2], [1.0, 2.0], atol=1e-6)
    assert np.allclose(advantages[3], [3.0, 4.0], atol=1e-6)
