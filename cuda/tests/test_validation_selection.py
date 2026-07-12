"""Validation selection and CPU orchestration checks."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from estimation_calibration_cuda.batched_calibration import train_batched
from estimation_calibration_cuda.covariance_calibration import (
    CalibrationConfig,
    trajectory_metrics,
    validate_training_splits,
)
from estimation_calibration_cuda.data import _episode_to_rollout, load_dataset


def _example_rollouts(split: str):
    episodes = load_dataset("example").load(split)
    rollouts = {
        episode.name: _episode_to_rollout(episode, device=torch.device("cpu"))
        for episode in episodes
    }
    return list(rollouts), rollouts


def _config(epochs=1):
    return CalibrationConfig(
        trim_s=0.0,
        epochs=epochs,
        chunk=32,
        require_cuda=False,
        compile_mode=None,
        seed=17,
    )


def test_cpu_eager_validation_metrics_without_cuda_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected CUDA-only call")

    for name in ("reset_peak_memory_stats", "max_memory_allocated", "synchronize",
                 "get_device_name"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    train_order, train_rollouts = _example_rollouts("train")
    val_order, val_rollouts = _example_rollouts("validation")
    before = torch.get_default_dtype()
    result = train_batched(
        train_order, train_rollouts,
        validation_order=val_order, validation_rollouts=val_rollouts,
        config=_config(), device=torch.device("cpu"))
    metrics = result.history[0]["validation"]
    assert metrics["finite"]
    assert metrics["final_P_sym"] < 1e-9
    assert metrics["final_P_min_eig"] > -1e-12
    assert metrics["body_velocity_rmse_mps"] >= 0
    assert metrics["orientation_mean_deg"] >= 0
    assert metrics["position_rmse_m"] >= 0
    assert torch.get_default_dtype() == before


def test_validation_alone_selects_checkpoint():
    train_order, train_rollouts = _example_rollouts("train")
    val_order, val_rollouts = _example_rollouts("validation")
    states = []
    validation_curve = [0.3, 0.1, 0.2]

    def validation(modules, epoch):
        states.append(copy.deepcopy(modules.state_dict()))
        return {"body_velocity_rmse_mps": validation_curve[epoch]}

    result = train_batched(
        train_order, train_rollouts,
        validation_order=val_order, validation_rollouts=val_rollouts,
        config=_config(epochs=3), device=torch.device("cpu"),
        validation_callback=validation)
    assert result.best["epoch"] == 1
    assert np.argmin([row["train_body_loss"] for row in result.history]) != 1
    for key, value in result.modules.state_dict().items():
        assert torch.equal(value, states[1][key])


@pytest.mark.parametrize("case", ["missing_train", "missing_validation", "overlap"])
def test_invalid_training_splits_fail(case):
    train_order, train_rollouts = _example_rollouts("train")
    val_order, val_rollouts = _example_rollouts("validation")
    if case == "missing_train":
        train_order, train_rollouts = [], {}
    elif case == "missing_validation":
        val_order, val_rollouts = [], {}
    else:
        val_order = train_order.copy()
        val_rollouts = train_rollouts.copy()
    with pytest.raises(ValueError, match="requires|overlap"):
        validate_training_splits(
            train_order, train_rollouts, val_order, val_rollouts)


def test_loading_training_splits_never_opens_test(monkeypatch):
    calls = []
    original = np.load

    def recording_load(path, *args, **kwargs):
        calls.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    dataset = load_dataset("example")
    dataset.load("train")
    dataset.load("validation")
    assert calls == ["train.npz", "validation.npz"]
    assert "test.npz" not in calls


def test_metric_units_and_nis_definition():
    R = torch.eye(3, dtype=torch.float64).expand(3, 3, 3).clone()
    zeros = torch.zeros(3, 3, dtype=torch.float64)
    metrics = trajectory_metrics(
        R, zeros, zeros, R, zeros, zeros,
        torch.eye(15, dtype=torch.float64),
        nis=torch.tensor([3.0, 6.0], dtype=torch.float64),
        nis_dim=torch.tensor([3.0, 3.0], dtype=torch.float64),
    )
    assert metrics["body_velocity_rmse_mps"] == 0
    assert metrics["orientation_mean_deg"] == 0
    assert metrics["orientation_max_deg"] == 0
    assert metrics["position_rmse_m"] == 0
    assert metrics["position_final_error_m"] == 0
    assert metrics["nis_per_measurement_dim"] == 1.5
