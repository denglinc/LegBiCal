"""External-data-free end-to-end replay golden."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from estimation_calibration_cuda import fixed_slot_inekf as fsi
from estimation_calibration_cuda.invariant_ekf import replay_inekf_torch

from conftest import (
    SYNTHETIC_GOLDEN,
    dynamic_column_map,
    map_dynamic_P_to_fixed,
)


EXPECTED_SHA256 = "be697fb80b8eec90085562487c9d5de343e97e6ac1eb16a60e7f69636c998002"


def _tensor(data, key):
    return torch.as_tensor(data[key], dtype=torch.float64)


def _fixed_inputs(data):
    total_rows, candidates = data["flags"].shape
    flags = np.zeros((total_rows, fsi.N_SLOTS), dtype=bool)
    flags[:, :candidates] = data["flags"]
    p_meas = torch.zeros(total_rows, fsi.N_SLOTS, 3, dtype=torch.float64)
    p_meas[:, :candidates] = _tensor(data, "p_meas")
    imu = torch.zeros(total_rows, 6, dtype=torch.float64)
    imu[1:] = _tensor(data, "imu")[:-1]
    previous = np.zeros_like(flags)
    previous[1:] = flags[:-1]
    correcting = previous & flags
    mask = lambda value: torch.as_tensor(value)[None]
    batch = fsi.BatchData(
        B=1,
        T_pad=total_rows,
        imu=imu[None],
        p_meas=p_meas[None],
        gt_R_WB=torch.eye(3, dtype=torch.float64).expand(
            1, total_rows, 3, 3).clone(),
        gt_v_B=torch.zeros(1, total_rows, 3, dtype=torch.float64),
        gt_p_W=torch.zeros(1, total_rows, 3, dtype=torch.float64),
        dt_row=torch.full((1, total_rows), float(data["dt"]),
                          dtype=torch.float64),
        valid=torch.ones(1, total_rows, dtype=torch.bool),
        prop_mask=mask(previous),
        correct_mask=mask(correcting),
        insert_mask=mask(~previous & flags),
        nis_dim=3.0 * mask(correcting).sum(-1).to(torch.float64),
    )
    batch.dt_row[:, 0] = 0.0
    return flags, batch


def test_synthetic_fixture_integrity():
    assert hashlib.sha256(SYNTHETIC_GOLDEN.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_dynamic_and_fixed_match_synthetic_golden(synthetic_golden):
    data = synthetic_golden
    covariances = {
        key: _tensor(data, key) for key in ("Qg", "Qa", "Qbg", "Qba", "Qc")
    }
    R_kin = _tensor(data, "R_kin")
    with torch.no_grad():
        dynamic = replay_inekf_torch(
            _tensor(data, "X0"),
            _tensor(data, "theta0"),
            _tensor(data, "P0"),
            covariances,
            _tensor(data, "imu"),
            float(data["dt"]),
            _tensor(data, "p_meas"),
            data["flags"],
            R_kin,
            collect_nis=True,
        )

    for key, reference in (
        ("R_WB", "R_np"), ("v_W", "v_np"), ("p_W", "p_np")
    ):
        assert torch.allclose(
            dynamic[key], _tensor(data, reference), rtol=0.0, atol=1e-12
        )
    for key in ("final_X", "final_theta", "final_P"):
        assert torch.allclose(
            dynamic[key], _tensor(data, key), rtol=0.0, atol=1e-12
        )
    expected_map = dict(zip(
        data["final_contact_ids"].tolist(),
        data["final_contact_cols"].tolist(),
    ))
    assert dynamic["final_estimated_contact_positions"] == expected_map

    flags, batch = _fixed_inputs(data)
    with torch.no_grad():
        state = fsi.init_state(
            [(_tensor(data, "X0"), _tensor(data, "theta0"),
              _tensor(data, "P0"))],
            device=torch.device("cpu"),
        )
        state = fsi.apply_row0(
            state, batch.p_meas[:, 0], batch.insert_mask[:, 0], R_kin
        )
        row0 = {
            "R_WB": state.X[:, None, 0:3, 0:3],
            "v_W": state.X[:, None, 0:3, 3],
            "p_W": state.X[:, None, 0:3, 4],
        }
        state, fixed_tail = fsi.run_rows_fixed(
            state, batch, slice(1, batch.T_pad), covariances, R_kin
        )

    for key, reference in (
        ("R_WB", "R_np"), ("v_W", "v_np"), ("p_W", "p_np")
    ):
        fixed = torch.cat([row0[key], fixed_tail[key]], dim=1)[0]
        assert torch.allclose(
            fixed, _tensor(data, reference), rtol=0.0, atol=1e-12
        )

    dynamic_nis = torch.stack(dynamic["nis_values"])
    fixed_nis = fixed_tail["nis"][0][fixed_tail["nis_dim"][0] > 0]
    assert torch.allclose(fixed_nis, dynamic_nis, rtol=0.0, atol=1e-12)
    positions = dynamic_column_map(flags)
    dynamic_P, fixed_P = map_dynamic_P_to_fixed(
        dynamic["final_P"], positions, state.P[0]
    )
    assert torch.allclose(fixed_P, dynamic_P, rtol=0.0, atol=1e-12)
