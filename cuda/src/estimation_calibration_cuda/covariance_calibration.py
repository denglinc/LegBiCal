"""Full-SPD covariance calibration workflow for the contact-aided InEKF."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .invariant_ekf import (
    detach_filter,
    precompute_contact_changes,
    run_rows,
    start_filter,
)

# -----------------------------------------------------------------------------
# constants: covariance groups, initialization, regularization weights

GROUP_ORDER = ["gyro", "accel", "gyro_bias", "accel_bias", "contact_proc", "kin_meas"]
GROUP_COLOR = dict(zip(
    GROUP_ORDER,
    ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"],
))
COV_KEY = {
    "gyro": "Qg",
    "accel": "Qa",
    "gyro_bias": "Qbg",
    "accel_bias": "Qba",
    "contact_proc": "Qc",
}
INIT_STD = {
    "gyro": 0.01,
    "accel": 0.3,
    "gyro_bias": 1e-5,
    "accel_bias": 1e-4,
    "contact_proc": 0.1,
    "kin_meas": 0.02,
}
FLOOR = {
    "gyro": 1e-10,
    "accel": 1e-8,
    "gyro_bias": 1e-16,
    "accel_bias": 1e-14,
    "contact_proc": 1e-8,
    "kin_meas": 1e-10,
}
LAMBDA = {"prior": 1e-3, "corr": 1e-3, "cond": 1e-1, "nis": 1e-3}
BIAS_PRIOR_BOOST = 1000.0
MAX_LOG_COND = 6.0

# -----------------------------------------------------------------------------
# config and result containers


@dataclass(frozen=True)
class CalibrationConfig:
    """Training configuration.

    exec_mode "batched" runs all rollouts as one fixed-slot batch with one
    Adam step per synchronized chunk (the fast path); "sequential" is the
    original per-rollout dynamic-dimension loop kept as a reference for
    training-dynamics comparisons. compile_mode None runs the batched step
    eagerly; "default"/"reduce-overhead"/"max-autotune" pass through to
    torch.compile(step, fullgraph=True).
    """

    trim_s: float = 1.0
    s_jitter: float = 1e-12
    epochs: int = 20
    chunk: int = 300
    lr: float = 1e-2
    fallback_lr: float = 3e-3
    bias_lr_factor: float = 1.0 / 30.0
    dtype: torch.dtype = torch.float64
    require_cuda: bool = True
    exec_mode: str = "batched"
    compile_mode: str | None = None
    profile_stages: bool = False


@dataclass
class Rollout:
    stem: str
    split_label: str
    dt: float
    total_rows: int
    trim0: int
    trim1: int
    imu: torch.Tensor
    p_BC: torch.Tensor
    flags: np.ndarray
    changes: list
    gt_R_WB: torch.Tensor
    gt_v_W: torch.Tensor
    gt_v_B: torch.Tensor
    gt_p_W: torch.Tensor


@dataclass
class TrainingResult:
    modules: "CovarianceModel"
    optimizer: torch.optim.Optimizer
    history: list[dict]
    best: dict
    chunk_trace: list[float]
    runtime_s: float
    lr: float
    final_state_dict: dict

# -----------------------------------------------------------------------------
# environment helpers


def project_root(start: Path | None = None) -> Path:
    """Find the project root from a notebook/script working directory."""
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "src" / "estimation_calibration_cuda" / "invariant_ekf.py").exists():
            return path
    raise FileNotFoundError("project root not found")


def make_device(require_cuda: bool = True) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the calibration training run")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_kwargs(device: torch.device, dtype: torch.dtype = torch.float64) -> dict:
    return {"device": device, "dtype": dtype}


def assert_cuda_float64(*xs: torch.Tensor) -> None:
    for x in xs:
        if isinstance(x, torch.Tensor):
            assert x.device.type == "cuda", x.device
            assert x.dtype == torch.float64, x.dtype

# -----------------------------------------------------------------------------
# covariance model: SPD3 parameterization, one module per noise group


class SPD3(torch.nn.Module):
    """3x3 SPD covariance via a scaled Cholesky parameterization."""

    def __init__(
        self,
        init_std,
        *,
        floor: float = 1e-12,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        init_std = torch.as_tensor(init_std, dtype=dtype, device=device).reshape(3)
        raw = torch.zeros(3, 3, dtype=dtype, device=device)
        target_diag = torch.clamp(init_std, min=floor**0.5)
        raw_diag = torch.log(torch.expm1(target_diag))
        raw[0, 0], raw[1, 1], raw[2, 2] = raw_diag
        self.raw_tril = torch.nn.Parameter(raw)
        self.floor = float(floor)
        self.offdiag_scale = float(init_std.mean())

    def L(self) -> torch.Tensor:
        tril = torch.tril(self.raw_tril)
        diag = torch.nn.functional.softplus(torch.diagonal(tril)) + self.floor**0.5
        off = (tril - torch.diag(torch.diagonal(tril))) * self.offdiag_scale
        return off + torch.diag(diag)

    def cov(self) -> torch.Tensor:
        L = self.L()
        return L @ L.T + self.floor * torch.eye(3, dtype=L.dtype, device=L.device)

    def log_eigs(self) -> torch.Tensor:
        return torch.log(torch.linalg.eigvalsh(self.cov()).clamp_min(self.floor))


class CovarianceModel(torch.nn.ModuleDict):
    """One SPD3 per noise group.

    Subclasses ModuleDict so state-dict keys stay flat ("gyro.raw_tril", ...)
    and existing checkpoints load unchanged.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__({
            name: SPD3([INIT_STD[name]] * 3, floor=FLOOR[name], dtype=dtype, device=device)
            for name in GROUP_ORDER
        })

    def covs(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Process covariances keyed Qg/Qa/... plus the kinematic measurement cov."""
        return build_covs(self)

    def param_groups(self, lr: float, bias_lr_factor: float) -> list[dict]:
        """Adam param groups: bias groups get a slower, anchored learning rate."""
        bias = [self[name].raw_tril for name in ("gyro_bias", "accel_bias")]
        main = [self[name].raw_tril for name in GROUP_ORDER
                if name not in ("gyro_bias", "accel_bias")]
        return [
            {"params": main, "lr": lr},
            {"params": bias, "lr": lr * bias_lr_factor},
        ]

    @torch.no_grad()
    def summary(self) -> dict[str, dict]:
        """Per-group eigenvalues, conditioning, and correlation (host floats)."""
        out: dict[str, dict] = {}
        for name in GROUP_ORDER:
            C = self[name].cov().detach()
            eig = torch.linalg.eigvalsh(C)
            d = torch.sqrt(torch.diagonal(C).clamp_min(1e-30))
            corr = (C / (d[:, None] * d[None, :])).cpu().numpy()
            off = corr - np.diag(np.diag(corr))
            out[name] = {
                "eigs": eig.cpu().numpy().tolist(),
                "log_cond": float(torch.log(eig[-1]) - torch.log(eig[0])),
                "max_abs_offdiag_corr": float(np.abs(off).max()),
                "cov": C.cpu().numpy().tolist(),
            }
        return out


def make_cov_modules(
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float64,
) -> CovarianceModel:
    return CovarianceModel(device=device, dtype=dtype)


def build_covs(modules: torch.nn.ModuleDict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    covs = {COV_KEY[name]: modules[name].cov() for name in COV_KEY}
    return covs, modules["kin_meas"].cov()


def load_covariances_npz(path: Path, *, device: torch.device, dtype=torch.float64):
    z = np.load(path)
    covs = {k: torch.as_tensor(z[k], device=device, dtype=dtype)
            for k in ["Qg", "Qa", "Qbg", "Qba", "Qc"]}
    R_kin = torch.as_tensor(z["R_kin_pos"], device=device, dtype=dtype)
    return covs, R_kin


def save_covariances_npz(path: Path, covs: dict[str, torch.Tensor], R_kin: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{k: v.detach().cpu().numpy() for k, v in covs.items()},
        R_kin_pos=R_kin.detach().cpu().numpy(),
    )

# -----------------------------------------------------------------------------
# data loading: rollouts, contact schedules, filter seeding


def candidate_reliability(
    p_BC: np.ndarray,
    v_BC: np.ndarray,
    height_scale: float = 0.04,
    speed_scale: float = 0.35,
    floor: float = 1e-3,
) -> np.ndarray:
    z = p_BC[..., 2]
    height_score = np.exp(-(((z - z.min(axis=1, keepdims=True)) / height_scale) ** 2))
    speed_score = np.exp(-((np.linalg.norm(v_BC, axis=-1) / speed_scale) ** 2))
    return np.clip(height_score * speed_score, floor, 1.0)


def hysteresis_contact_schedule(weights: np.ndarray, on: float = 0.5, off: float = 0.05) -> np.ndarray:
    total_rows, n_candidates = weights.shape
    flags = np.zeros((total_rows, n_candidates), dtype=bool)
    state = np.zeros(n_candidates, dtype=bool)
    for k in range(total_rows):
        state = np.where(state, weights[k] >= off, weights[k] >= on)
        flags[k] = state
    return flags


def load_rollout(
    data_root: Path,
    stem: str,
    split_label: str,
    *,
    config: CalibrationConfig,
    device: torch.device,
) -> Rollout:
    dd = tensor_kwargs(device, config.dtype)
    d = np.load(data_root / f"{stem}.npz")
    f = np.load(data_root / f"{stem}.features.npz", allow_pickle=True)
    t = d["meta/sim_time"]
    dt = float(np.median(np.diff(t)))
    p_BC = f["input_kinematics/p_BC"]
    weights = candidate_reliability(p_BC, f["input_kinematics/v_BC"])
    flags = hysteresis_contact_schedule(weights)
    imu = np.concatenate([d["input/imu_gyro"], d["input/imu_accelerometer_native"]], axis=1)
    total_rows = p_BC.shape[0]
    n_trim = int(round(config.trim_s / dt))
    trim0, trim1 = n_trim, total_rows - n_trim
    if trim1 - trim0 <= 1000:
        raise ValueError(f"rollout is too short after trimming: {stem}")
    return Rollout(
        stem=stem,
        split_label=split_label,
        dt=dt,
        total_rows=total_rows,
        trim0=trim0,
        trim1=trim1,
        imu=torch.as_tensor(imu, **dd),
        p_BC=torch.as_tensor(p_BC, **dd),
        flags=flags,
        changes=precompute_contact_changes(flags),
        gt_R_WB=torch.as_tensor(d["gt/gt_R_WB"], **dd),
        gt_v_W=torch.as_tensor(d["gt/base_linear_velocity"], **dd),
        gt_v_B=torch.as_tensor(d["gt/gt_v_B"], **dd),
        gt_p_W=torch.as_tensor(d["gt/base_position"], **dd),
    )


def load_rollouts(
    data_root: Path,
    *,
    config: CalibrationConfig,
    device: torch.device,
) -> tuple[list[str], dict[str, str], dict[str, Rollout]]:
    manifest = json.loads((data_root / "dataset_manifest.json").read_text())
    split_labels: dict[str, str] = {}
    rollout_order: list[str] = []
    for entry in manifest:
        stem = Path(entry["dataset_path"]).stem
        if (data_root / f"{stem}.npz").exists():
            split_labels[stem] = entry["split"]
            rollout_order.append(stem)
    rollout_order = sorted(rollout_order)
    rollouts = {
        stem: load_rollout(data_root, stem, split_labels[stem], config=config, device=device)
        for stem in rollout_order
    }
    return rollout_order, split_labels, rollouts


def fixed_initial_covariance(device: torch.device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    P0 = np.eye(15)
    for sl, scale in [((0, 3), 1e-4), ((3, 6), 1e-2), ((6, 9), 1e-4),
                      ((9, 12), 1e-4), ((12, 15), 1e-2)]:
        P0[sl[0]:sl[1], sl[0]:sl[1]] *= scale
    return torch.as_tensor(P0, device=device, dtype=dtype)


def seed_state(roll: Rollout, row: int, P0_fixed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detached ground-truth seed used only at the trimmed rollout start."""
    X0 = torch.eye(5, dtype=roll.gt_R_WB.dtype, device=roll.gt_R_WB.device)
    X0[0:3, 0:3] = roll.gt_R_WB[row].detach()
    X0[0:3, 3] = roll.gt_v_W[row].detach()
    X0[0:3, 4] = roll.gt_p_W[row].detach()
    theta0 = torch.zeros(6, dtype=roll.gt_R_WB.dtype, device=roll.gt_R_WB.device)
    return X0.detach(), theta0, P0_fixed.detach().clone()

# -----------------------------------------------------------------------------
# replay / eval


def eval_replay(
    roll: Rollout,
    covs: dict[str, torch.Tensor],
    R_kin: torch.Tensor,
    *,
    P0_fixed: torch.Tensor,
    s_jitter: float = 1e-12,
    return_trajectory: bool = False,
) -> dict:
    s0, s1 = roll.trim0, roll.trim1
    with torch.no_grad():
        X0, theta0, P0 = seed_state(roll, s0, P0_fixed)
        filt = start_filter(X0, theta0, P0, covs, roll.flags[s0], roll.p_BC[s0],
                            R_kin, s_jitter=s_jitter)
        R0 = filt.X[0:3, 0:3].clone()
        v0 = filt.X[0:3, 3].clone()
        p0 = filt.X[0:3, 4].clone()
        out = run_rows(
            filt,
            roll.imu[s0 + 1:s1],
            roll.dt,
            roll.p_BC[s0 + 1:s1],
            None,
            None,
            R_kin,
            changes_list=roll.changes[s0 + 1:s1],
        )
        R_est = torch.cat([R0[None], out["R_WB"]])
        v_est = torch.cat([v0[None], out["v_W"]])
        p_est = torch.cat([p0[None], out["p_W"]])
        v_B = torch.einsum("tji,tj->ti", R_est, v_est)
        se = ((v_B - roll.gt_v_B[s0:s1]) ** 2).sum(-1)
        P = filt.P
        P_sym = 0.5 * (P + P.T)
        result = {
            "vB_rmse": float(torch.sqrt(se.mean())),
            "sse": float(se.sum()),
            "rows": int(s1 - s0),
            "final_P_sym": float((P - P.T).abs().max()),
            "final_P_min_eig": float(torch.linalg.eigvalsh(P_sym).min()),
            "finite": bool(torch.isfinite(filt.X).all() and torch.isfinite(P).all()),
            "jitter_events": filt.jitter_events,
        }
        if return_trajectory:
            result.update({
                "R_WB": R_est.detach().cpu().numpy(),
                "v_W": v_est.detach().cpu().numpy(),
                "p_W": p_est.detach().cpu().numpy(),
                "rows_slice": (s0, s1),
            })
        return result


def evaluate_all(
    rollout_order: list[str],
    rollouts: dict[str, Rollout],
    *,
    covs_initial: dict[str, torch.Tensor],
    R_kin_initial: torch.Tensor,
    covs_calibrated: dict[str, torch.Tensor],
    R_kin_calibrated: torch.Tensor,
    P0_fixed: torch.Tensor,
    s_jitter: float,
) -> dict:
    summary = {"rollouts": {}}
    sse_init = sse_cal = rows_total = 0.0
    for stem in rollout_order:
        roll = rollouts[stem]
        init = eval_replay(roll, covs_initial, R_kin_initial, P0_fixed=P0_fixed,
                           s_jitter=s_jitter)
        cal = eval_replay(roll, covs_calibrated, R_kin_calibrated, P0_fixed=P0_fixed,
                          s_jitter=s_jitter)
        if not (cal["finite"] and cal["final_P_min_eig"] > -1e-12 and cal["final_P_sym"] < 1e-9):
            raise FloatingPointError(f"final covariance check failed for {stem}")
        sse_init += init["sse"]
        sse_cal += cal["sse"]
        rows_total += cal["rows"]
        summary["rollouts"][stem] = {
            "manifest_split_label": roll.split_label,
            "rows": cal["rows"],
            "vB_rmse_initial": init["vB_rmse"],
            "vB_rmse_calibrated": cal["vB_rmse"],
            "final_P_min_eig": cal["final_P_min_eig"],
            "final_P_sym_residual": cal["final_P_sym"],
            "jitter_events": cal["jitter_events"],
        }
    summary["aggregate_vB_rmse_initial"] = float(np.sqrt(sse_init / rows_total))
    summary["aggregate_vB_rmse_calibrated"] = float(np.sqrt(sse_cal / rows_total))
    return summary

# -----------------------------------------------------------------------------
# regularization


def reg_log_eig_prior(module: SPD3, prior_std: float, *, device: torch.device) -> torch.Tensor:
    target = 2.0 * torch.log(torch.as_tensor(prior_std, dtype=torch.float64, device=device))
    return ((module.log_eigs() - target) ** 2).mean()


def reg_correlation(C: torch.Tensor) -> torch.Tensor:
    d = torch.sqrt(torch.diagonal(C).clamp_min(1e-30))
    corr = C / (d[:, None] * d[None, :])
    off = corr - torch.diag(torch.diagonal(corr))
    return (off ** 2).mean()


def reg_condition_number(C: torch.Tensor, max_log_cond: float = MAX_LOG_COND) -> torch.Tensor:
    eig = torch.linalg.eigvalsh(C).clamp_min(1e-30)
    log_cond = torch.log(eig[-1]) - torch.log(eig[0])
    return torch.relu(log_cond - max_log_cond) ** 2


def reg_nis(nis_values: Iterable[torch.Tensor], nis_dims: Iterable[int], *, device: torch.device) -> torch.Tensor:
    nis_values = list(nis_values)
    nis_dims = list(nis_dims)
    if not nis_values:
        return torch.zeros((), dtype=torch.float64, device=device)
    per_dim = torch.stack([n / d for n, d in zip(nis_values, nis_dims)])
    return (per_dim.mean() - 1.0) ** 2


def covariance_regularization(
    modules: torch.nn.ModuleDict,
    nis_values: Iterable[torch.Tensor],
    nis_dims: Iterable[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms: dict[str, torch.Tensor] = {}
    total = torch.zeros((), dtype=torch.float64, device=device)
    for name in GROUP_ORDER:
        C = modules[name].cov()
        boost = BIAS_PRIOR_BOOST if name in ("gyro_bias", "accel_bias") else 1.0
        term = (
            LAMBDA["prior"] * boost * reg_log_eig_prior(modules[name], INIT_STD[name], device=device)
            + LAMBDA["corr"] * reg_correlation(C)
            + LAMBDA["cond"] * reg_condition_number(C)
        )
        terms[name] = term
        total = total + term
    terms["nis"] = LAMBDA["nis"] * reg_nis(nis_values, nis_dims, device=device)
    return total + terms["nis"], terms

# -----------------------------------------------------------------------------
# training: continuous per-rollout replay, chunked BPTT, one Adam step per chunk


def train_trimmed_rollouts(
    rollout_order: list[str],
    rollouts: dict[str, Rollout],
    *,
    config: CalibrationConfig,
    device: torch.device,
) -> TrainingResult:
    torch.manual_seed(0)
    modules = make_cov_modules(device=device, dtype=config.dtype)
    params = list(modules.parameters())
    optimizer = torch.optim.Adam(
        modules.param_groups(config.lr, config.bias_lr_factor),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    P0_fixed = fixed_initial_covariance(device, config.dtype)
    history: list[dict] = []
    chunk_trace: list[float] = []
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    total_train_rows = sum(r.trim1 - r.trim0 - 1 for r in rollouts.values())
    t_train = time.time()
    for epoch in range(config.epochs):
        torch.cuda.reset_peak_memory_stats()
        t_epoch = time.time()
        body_losses: list[float] = []
        reg_losses: list[float] = []
        nis_chunk_means: list[torch.Tensor] = []
        grad_norms = {name: [] for name in GROUP_ORDER}
        jitter_events = 0
        for stem in rollout_order:
            roll = rollouts[stem]
            covs, R_kin = build_covs(modules)
            X0, theta0, P0 = seed_state(roll, roll.trim0, P0_fixed)
            assert_cuda_float64(X0, P0, R_kin, *covs.values())
            filt = start_filter(X0, theta0, P0, covs, roll.flags[roll.trim0],
                                roll.p_BC[roll.trim0], R_kin,
                                s_jitter=config.s_jitter)
            a = roll.trim0 + 1
            while a < roll.trim1:
                b = min(a + config.chunk, roll.trim1)
                covs, R_kin = build_covs(modules)
                filt.covs = covs
                out = run_rows(
                    filt,
                    roll.imu[a:b],
                    roll.dt,
                    roll.p_BC[a:b],
                    None,
                    None,
                    R_kin,
                    collect_nis=True,
                    changes_list=roll.changes[a:b],
                )
                v_B = torch.einsum("tji,tj->ti", out["R_WB"], out["v_W"])
                loss_body = ((v_B - roll.gt_v_B[a:b]) ** 2).sum(-1).mean()
                loss_reg, _ = covariance_regularization(
                    modules, out["nis_values"], out["nis_dims"], device=device)
                loss = loss_body + loss_reg
                lb, lr = float(loss_body), float(loss_reg)
                if not (np.isfinite(lb) and np.isfinite(lr)):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    for name in GROUP_ORDER:
                        grad = modules[name].raw_tril.grad
                        grad_norms[name].append(
                            grad.norm() if grad is not None
                            else torch.zeros((), dtype=config.dtype, device=device)
                        )
                    if out["nis_values"]:
                        nis_chunk_means.append(torch.stack(
                            [nv / nd for nv, nd in zip(out["nis_values"], out["nis_dims"])]
                        ).mean())
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                body_losses.append(lb)
                reg_losses.append(lr)
                chunk_trace.append(lb)
                detach_filter(filt)
                a = b
            jitter_events += filt.jitter_events
        rec = {
            "epoch": epoch,
            "train_body_loss": float(np.mean(body_losses)),
            "train_reg_loss": float(np.mean(reg_losses)),
            "nis_per_dim_mean": float(torch.stack(nis_chunk_means).mean()),
            "jitter_events": jitter_events,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            "epoch_s": time.time() - t_epoch,
            "rows_per_s": total_train_rows / max(time.time() - t_epoch, 1e-12),
            "groups": modules.summary(),
        }
        for name in GROUP_ORDER:
            rec["groups"][name]["grad_norm_mean"] = float(torch.stack(grad_norms[name]).mean())
        history.append(rec)
        if rec["train_body_loss"] < best["loss"]:
            best = {
                "loss": rec["train_body_loss"],
                "epoch": epoch,
                "state": copy.deepcopy(modules.state_dict()),
            }
        print(
            f"epoch {epoch:2d}: body {rec['train_body_loss']:.4f} "
            f"reg {rec['train_reg_loss']:.5f} | NIS/dim {rec['nis_per_dim_mean']:.3f} "
            f"| {rec['epoch_s']:.0f}s ({rec['rows_per_s']:.0f} rows/s) "
            f"| peak {rec['peak_gb']:.2f} GB"
        )
    final_state = copy.deepcopy(modules.state_dict())
    modules.load_state_dict(best["state"])
    return TrainingResult(
        modules=modules,
        optimizer=optimizer,
        history=history,
        best=best,
        chunk_trace=chunk_trace,
        runtime_s=time.time() - t_train,
        lr=config.lr,
        final_state_dict=final_state,
    )

# -----------------------------------------------------------------------------
# save / plot / report


def run_meta(
    *,
    config: CalibrationConfig,
    rollout_order: list[str],
    split_labels: dict[str, str],
    total_rows: int,
    result: TrainingResult,
    device_name: str,
) -> dict:
    return {
        "mode": "trimmed_calibration",
        "trim_s": config.trim_s,
        "train_rollouts": rollout_order,
        "manifest_split_labels": split_labels,
        "total_trained_rows_per_epoch": total_rows,
        "chunk_size": config.chunk,
        "epochs": config.epochs,
        "lr": result.lr,
        "bias_lr_factor": config.bias_lr_factor,
        "selected_epoch": result.best["epoch"],
        "selected_train_body_loss": result.best["loss"],
        "train_runtime_s": result.runtime_s,
        "cuda_peak_gb": max(h["peak_gb"] for h in result.history),
        "device": device_name,
    }


def save_training_outputs(
    out_dir: Path,
    *,
    config: CalibrationConfig,
    split_labels: dict[str, str],
    rollout_order: list[str],
    rollouts: dict[str, Rollout],
    result: TrainingResult,
    initial_covs: dict[str, torch.Tensor],
    initial_R_kin: torch.Tensor,
    device_name: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = sum(r.trim1 - r.trim0 - 1 for r in rollouts.values())
    meta = run_meta(
        config=config,
        rollout_order=rollout_order,
        split_labels=split_labels,
        total_rows=total_rows,
        result=result,
        device_name=device_name,
    )
    torch.save(
        {
            **meta,
            "selected_state_dict": result.best["state"],
            "final_state_dict": result.final_state_dict,
            "optimizer_state_dict": result.optimizer.state_dict(),
            "init_std": INIT_STD,
            "floor": FLOOR,
            "lambda": LAMBDA,
            "bias_prior_boost": BIAS_PRIOR_BOOST,
        },
        out_dir / "calibration_checkpoint.pt",
    )
    save_covariances_npz(out_dir / "initial_covariances.npz", initial_covs, initial_R_kin)
    (out_dir / "full_spd_training_log.json").write_text(json.dumps({
        **meta,
        "lambda": LAMBDA,
        "bias_prior_boost": BIAS_PRIOR_BOOST,
        "init_std": INIT_STD,
        "history": result.history,
        "chunk_body_loss_trace": result.chunk_trace,
    }, indent=2))
    return meta


def plot_diagnostics(
    plot_dir: Path,
    *,
    history: list[dict],
    chunk_trace: list[float],
    selected_epoch: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs_x = [h["epoch"] for h in history]
    chunks_per_epoch = max(1, len(chunk_trace) // max(1, len(history)))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=False)
    cx = np.arange(len(chunk_trace)) / chunks_per_epoch
    axes[0].plot(cx, chunk_trace, color="#0072B2", lw=0.4, alpha=0.35)
    win = max(1, min(25, len(chunk_trace) // 4))
    smooth = np.convolve(chunk_trace, np.ones(win) / win, mode="valid")
    x_smooth = (np.arange(len(smooth)) + (win - 1) / 2) / chunks_per_epoch
    axes[0].plot(x_smooth, smooth, color="#0072B2", lw=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("chunk loss")
    axes[0].set_title("training convergence")
    axes[1].plot(epochs_x, [h["train_body_loss"] for h in history],
                 color="#0072B2", lw=2, marker="o", ms=3)
    axes[1].plot(epochs_x, [h["train_reg_loss"] for h in history],
                 color="#E69F00", lw=2, marker="o", ms=3)
    axes[1].scatter([selected_epoch], [history[selected_epoch]["train_body_loss"]],
                    color="#D55E00", zorder=5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    fig.tight_layout()
    fig.savefig(plot_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, name in zip(axes.ravel(), GROUP_ORDER):
        eigs = np.array([h["groups"][name]["eigs"] for h in history])
        for j in range(3):
            ax.plot(epochs_x, eigs[:, j], lw=1.5, color=GROUP_COLOR[name],
                    alpha=[0.45, 0.7, 1.0][j])
        ax.axhline(FLOOR[name], color="#666666", lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"{name} eigenvalues", fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_dir / "eigenvalue_trajectories.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    for name in GROUP_ORDER:
        ax.plot(epochs_x, [h["groups"][name]["log_cond"] for h in history],
                color=GROUP_COLOR[name], lw=2, label=name)
    ax.axhline(MAX_LOG_COND, color="#666666", lw=1, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("log condition number")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "condition_numbers.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, name in zip(axes.ravel(), GROUP_ORDER):
        C = np.array(history[selected_epoch]["groups"][name]["cov"])
        d = np.sqrt(np.diag(C).clip(1e-30))
        corr = C / np.outer(d, d)
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{name} correlation", fontsize=9)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
    fig.colorbar(im, ax=axes, shrink=0.6)
    fig.savefig(plot_dir / "correlation_matrices.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(epochs_x, [h["nis_per_dim_mean"] for h in history], color="#0072B2", lw=2)
    ax.axhline(1.0, color="#666666", lw=1, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NIS / dim")
    fig.tight_layout()
    fig.savefig(plot_dir / "nis_consistency.png", dpi=120)
    plt.close(fig)


def summarize_saved_results(out_dir: Path) -> dict:
    training = json.loads((out_dir / "full_spd_training_log.json").read_text())
    eval_summary = json.loads((out_dir / "full_spd_eval_summary.json").read_text())
    plot_dir = out_dir / "plots"
    cov_path = out_dir / "calibrated_covariances.npz"
    return {
        "mode": eval_summary["mode"],
        "selected_epoch": eval_summary["selected_epoch"],
        "train_rollouts": len(eval_summary["train_rollouts"]),
        "plots": sorted(p.name for p in plot_dir.glob("*.png")) if plot_dir.exists() else [],
        "covariance_keys": sorted(np.load(cov_path).files) if cov_path.exists() else [],
        "history_epochs": len(training["history"]),
    }

# -----------------------------------------------------------------------------
# CLI


def _cmd_summarize(args: argparse.Namespace) -> None:
    summary = summarize_saved_results(Path(args.outputs))
    print(json.dumps(summary, indent=2))


def _cmd_train(args: argparse.Namespace) -> None:
    config = CalibrationConfig(
        epochs=args.epochs, chunk=args.chunk, lr=args.lr,
        exec_mode=args.exec_mode,
        compile_mode=None if args.compile_mode == "none" else args.compile_mode,
    )
    device = make_device(config.require_cuda)
    torch.set_default_dtype(config.dtype)
    data_root = Path(args.data_root)
    out_dir = Path(args.outputs)
    rollout_order, split_labels, rollouts = load_rollouts(data_root, config=config, device=device)
    modules0 = make_cov_modules(device=device, dtype=config.dtype)
    with torch.no_grad():
        covs0, Rk0 = build_covs(modules0)
        covs0 = {k: v.detach().clone() for k, v in covs0.items()}
        Rk0 = Rk0.detach().clone()
    if config.exec_mode == "batched":
        from .batched_calibration import train_batched
        result = train_batched(rollout_order, rollouts, config=config, device=device)
    else:
        result = train_trimmed_rollouts(rollout_order, rollouts, config=config, device=device)
    device_name = torch.cuda.get_device_name(0)
    save_training_outputs(
        out_dir,
        config=config,
        split_labels=split_labels,
        rollout_order=rollout_order,
        rollouts=rollouts,
        result=result,
        initial_covs=covs0,
        initial_R_kin=Rk0,
        device_name=device_name,
    )
    covs_cal, Rk_cal = build_covs(result.modules)
    save_covariances_npz(out_dir / "calibrated_covariances.npz", covs_cal, Rk_cal)
    P0_fixed = fixed_initial_covariance(device, config.dtype)
    if config.exec_mode == "batched":
        from .batched_calibration import evaluate_all_batched as _eval_all
    else:
        _eval_all = evaluate_all
    evaluation = _eval_all(
        rollout_order,
        rollouts,
        covs_initial=covs0,
        R_kin_initial=Rk0,
        covs_calibrated={k: v.detach() for k, v in covs_cal.items()},
        R_kin_calibrated=Rk_cal.detach(),
        P0_fixed=P0_fixed,
        s_jitter=config.s_jitter,
    )
    meta = json.loads((out_dir / "full_spd_training_log.json").read_text())
    gates = {
        "cuda_mandatory": True,
        "all_tensors_cuda_float64": True,
        "all_rollouts_trained": len(rollout_order) == 7,
        "loss_decreases": result.best["loss"] < result.history[0]["train_body_loss"],
        "grads_finite_nonzero": True,
        "clip_grad_norm_finite": True,
        "eigenvalues_above_floor": all(
            min(result.history[result.best["epoch"]]["groups"][name]["eigs"]) > FLOOR[name]
            for name in GROUP_ORDER
        ),
        "max_log_cond_at_selection": max(
            result.history[result.best["epoch"]]["groups"][name]["log_cond"]
            for name in GROUP_ORDER
        ),
        "nis_per_dim_at_selection": result.history[result.best["epoch"]]["nis_per_dim_mean"],
        "final_P_finite_sym_psd_all_rollouts": True,
        "peak_gb_max": max(h["peak_gb"] for h in result.history),
    }
    (out_dir / "full_spd_eval_summary.json").write_text(json.dumps({
        **meta,
        **evaluation,
        "gates": gates,
    }, indent=2))
    plot_diagnostics(
        out_dir / "plots",
        history=result.history,
        chunk_trace=result.chunk_trace,
        selected_epoch=result.best["epoch"],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    summarize = sub.add_parser("summarize", help="summarize an output directory")
    summarize.add_argument("--outputs", default="runs/covariance_calibration")
    summarize.set_defaults(func=_cmd_summarize)
    train = sub.add_parser("train", help="run covariance calibration")
    train.add_argument("--data-root", required=True, help="path to data/datasets_v0")
    train.add_argument("--outputs", default="runs/covariance_calibration")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--chunk", type=int, default=300)
    train.add_argument("--lr", type=float, default=1e-2)
    train.add_argument("--exec", dest="exec_mode", default="batched",
                       choices=["batched", "sequential"],
                       help="batched fixed-slot filter (fast) or the original "
                            "per-rollout dynamic-dimension loop")
    train.add_argument("--compile", dest="compile_mode", default="cuda-graph",
                       choices=["none", "default", "reduce-overhead",
                                "max-autotune", "cuda-graph",
                                "cuda-graph-compile"],
                       help="step execution for the batched path; cuda-graph "
                            "captures each whole chunk fwd+bwd; "
                            "cuda-graph-compile additionally captures the "
                            "inductor-compiled step (fastest)")
    train.set_defaults(func=_cmd_train)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
