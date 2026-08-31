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
    formal = json.loads(
        (PROJECT_ROOT / "config" / "formal_experiments.json").read_text(
            encoding="utf-8"
        )
    )
    assert MODULE.DEFAULT_CONTROLLER_VARIANT == "ours_full_corrected"
    assert formal["controller_variant"] == MODULE.DEFAULT_CONTROLLER_VARIANT
    assert formal["oracle_scores"] == "raw_rf_predictions_without_quality_penalty"


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


def test_preference_floor_projection_is_idempotent():
    integrator = MODULE.MultiExploreIntegrator.__new__(MODULE.MultiExploreIntegrator)
    integrator.preference_floor = 0.10
    once = integrator._normalize_weights(np.asarray([1.0, 0.0]))
    twice = integrator._normalize_weights(once)
    assert np.allclose(once, twice)


def test_v2_linear_ramp_preserves_early_exploration_and_reaches_full_strength():
    assert np.isclose(MODULE.linear_ramp(0.30, 0.30, 0.70), 0.0)
    assert np.isclose(MODULE.linear_ramp(0.50, 0.30, 0.70), 0.5)
    assert np.isclose(MODULE.linear_ramp(0.70, 0.30, 0.70), 1.0)
    assert np.isclose(MODULE.linear_ramp(0.10, 0.0, 0.0), 1.0)


def test_v2_mixed_preferences_are_stage_gated():
    integrator = MODULE.MultiExploreIntegrator.__new__(MODULE.MultiExploreIntegrator)
    integrator.sample_preference_mode = "grid"
    integrator.sample_preference_blend = 0.50
    integrator.preference_floor = 0.10
    base = np.asarray([0.65, 0.35], dtype=np.float32)
    early = integrator._build_sample_preferences(16, base, epoch=0, schedule_strength=0.0)
    late = integrator._build_sample_preferences(16, base, epoch=79, schedule_strength=1.0)
    assert np.allclose(early, np.repeat(base[None, :], 16, axis=0))
    assert np.std(late[:, 0]) > 0.05
    assert np.allclose(late.sum(axis=1), 1.0)
    assert np.all(late >= 0.10 - 1e-7)


def test_v21_hvc_balanced_archive_sampling_is_normalized_and_elite_biased():
    scores = np.asarray([[4.0, 8.0], [6.0, 6.0], [8.0, 4.0]], dtype=float)
    uniform = MODULE.archive_sampling_probabilities(scores, strategy="uniform")
    balanced = MODULE.archive_sampling_probabilities(
        scores,
        strategy="hvc_balanced",
        hvc_weight=0.7,
        balance_weight=0.3,
        temperature=0.25,
        uniform_mix=0.10,
    )
    assert np.allclose(uniform, np.ones(3) / 3)
    assert np.all(balanced > 0.0)
    assert np.isclose(balanced.sum(), 1.0)
    assert balanced[1] > balanced[0]
    assert balanced[1] > balanced[2]


def test_v21_archive_stagnation_gate_requires_full_window_and_small_gain():
    triggered, gain = MODULE.archive_stagnation_status([0.0, 0.10], 10, 0.002)
    assert not triggered and np.isnan(gain)
    improving = np.linspace(0.40, 0.42, 11)
    triggered, gain = MODULE.archive_stagnation_status(improving, 10, 0.002)
    assert not triggered
    assert np.isclose(gain, 0.02)
    stalled = np.linspace(0.40, 0.401, 11)
    triggered, gain = MODULE.archive_stagnation_status(stalled, 10, 0.002)
    assert triggered
    assert np.isclose(gain, 0.001)


def test_v3_generator_elite_ranking_matches_declared_balancing_rules():
    scores = np.asarray([[7.0, 4.0], [6.0, 6.0], [4.0, 7.0]], dtype=float)
    mean_rank = MODULE.generator_elite_scores(scores, "mean")
    min_rank = MODULE.generator_elite_scores(scores, "min")
    mixed_rank = MODULE.generator_elite_scores(scores, "mixed")
    assert mean_rank[1] > mean_rank[0]
    assert min_rank[1] > min_rank[0]
    assert mixed_rank[1] > mixed_rank[0]
    assert np.all((mean_rank >= 0.0) & (mean_rank <= 1.0))


def test_v4_raw_elite_ranking_preserves_activity_signal_above_6p5():
    scores = np.asarray([[6.6, 6.6], [8.0, 6.6], [8.0, 8.0]], dtype=float)
    clipped = MODULE.generator_elite_scores(scores, "mean")
    raw = MODULE.generator_elite_scores(scores, "raw_mean")
    assert np.allclose(clipped, clipped[0])
    assert raw[2] > raw[1] > raw[0]
    assert np.all((raw >= 0.0) & (raw <= 1.0))


def test_v5_balance_sync_prefers_dual_threshold_coverage():
    scores = np.asarray([[8.0, 6.0], [6.6, 6.6]], dtype=float)
    ranking = MODULE.generator_elite_scores(
        scores,
        "balance_sync",
        weights=np.asarray([0.5, 0.5]),
        balance_mix=0.5,
        softmin_temperature=0.1,
    )
    assert ranking[1] > ranking[0]


def test_v5_balance_sync_uses_live_controller_weights():
    scores = np.asarray([[6.4, 5.4], [5.4, 6.4]], dtype=float)
    favor_first = MODULE.generator_elite_scores(
        scores,
        "balance_sync",
        weights=np.asarray([0.8, 0.2]),
    )
    favor_second = MODULE.generator_elite_scores(
        scores,
        "balance_sync",
        weights=np.asarray([0.2, 0.8]),
    )
    assert favor_first[0] > favor_first[1]
    assert favor_second[1] > favor_second[0]


def test_v5_multi_critic_advantages_are_standardized_per_objective():
    agent = MODULE.TrajectoryMultiCriticPPOAgent(
        state_dim=4,
        action_dim=2,
        num_obj=2,
        mini_batch_size=1,
        device="cpu",
    )
    advantages = MODULE.torch.tensor(
        [[-1.0, -100.0], [0.0, 0.0], [1.0, 100.0]]
    )
    normalized = agent._standardize(advantages, dim=0)
    assert MODULE.torch.allclose(normalized.mean(dim=0), MODULE.torch.zeros(2))
    assert MODULE.torch.allclose(
        normalized.std(dim=0, unbiased=False), MODULE.torch.ones(2)
    )
    assert MODULE.torch.allclose(normalized[:, 0], normalized[:, 1])


def test_corrected_controller_does_not_mutate_phase3_during_phase1():
    controller = MODULE.create_controller(
        "ours_full_corrected", num_obj=2, total_epochs=160
    )
    controller.set_ref_point(np.asarray([3.0, 3.0]))
    molecules = [
        MODULE.Molecule("a", np.zeros(2), np.asarray([5.0, 3.2])),
        MODULE.Molecule("b", np.zeros(2), np.asarray([3.2, 5.0])),
        MODULE.Molecule("c", np.zeros(2), np.asarray([4.0, 4.2])),
    ]
    controller.update_pareto_front(molecules)
    controller.get_weights(0, np.asarray([[4.5, 4.1], [4.1, 4.5]]))
    assert controller._prev_weights is None
    assert not controller._weights_inherited_for_phase3


def test_phase2_feedback_is_continuous_for_two_objectives():
    controller = MODULE.create_controller(
        "ours_full_corrected", num_obj=2, total_epochs=160
    )
    controller.set_ref_point(np.asarray([3.0, 3.0]))
    controller.update_pareto_front([
        MODULE.Molecule("a", np.zeros(2), np.asarray([5.0, 3.2])),
        MODULE.Molecule("b", np.zeros(2), np.asarray([3.2, 5.0])),
        MODULE.Molecule("c", np.zeros(2), np.asarray([4.0, 4.2])),
    ])
    weights = controller._phase2_exploit(
        np.asarray([[5.4, 3.8], [4.7, 4.5]])
    )
    assert np.allclose(weights.sum(), 1.0)
    assert 0.15 < weights[0] < 0.85
    assert 0.15 < weights[1] < 0.85


def test_primary_canonicalization_keeps_only_largest_fragment():
    smiles, mol = MODULE.canonical("CCO.CCCCC")
    assert mol is not None
    assert smiles == "CCCCC"


def test_two_target_oracle_does_not_apply_quality_penalty():
    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, matrix):
            return np.full(len(matrix), self.value, dtype=np.float32)

    calculator = MODULE.TwoTargetObjectiveCalculator.__new__(
        MODULE.TwoTargetObjectiveCalculator
    )
    calculator.egfr_model = ConstantModel(5.25)
    calculator.vegfr2_model = ConstantModel(6.10)
    calculator._smiles_to_fingerprint = lambda smiles: np.zeros(2048, dtype=np.float32)
    long_smiles = "C" * 90
    single = calculator.calculate_scores(long_smiles)
    batch = calculator.calculate_scores_batch([long_smiles])[0]
    assert np.allclose(single, [5.25, 6.10])
    assert np.allclose(batch, [5.25, 6.10])


def test_registered_activity_hypervolume_is_normalized():
    assert np.isclose(MODULE.hypervolume_2d(np.asarray([[10.0, 10.0]])), 1.0)
    assert np.isclose(MODULE.hypervolume_2d(np.asarray([[3.0, 10.0]])), 0.0)


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
