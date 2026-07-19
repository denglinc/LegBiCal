"""Rendering entry points for the calibrated results.

Consumes saved lower-level solutions only (the shipped reference bundles or
a fresh solve's selected attempt); rendering never triggers a solve. The
published artifact is self-contained interactive Meshcat HTML. MuJoCo uses
the same frame contract through the passive interactive replay command.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .attempts import atomic_write_json
from .calibration import (
    CALIBRATION_STATEMENT,
    HORIZON_STATES,
    HORIZON_TRANSITIONS,
    component_specs,
)
from .gt_clips import load_gt_clip
from .loss import trajectory_loss_arrays
from .paths import project_root, resolve_inside_root
from .visualization.meshcat_system import record_meshcat_html
from .visualization.recording import (
    sampled_playback_fps,
    sampled_transition_indices,
    write_frame_metrics,
)
from .visualization.sequence import (
    MotionForceSequenceBuilder,
    load_visualization_style,
    write_sequence_artifacts,
)

DEFAULT_STYLE = "configs/visualization/default.yaml"
MESHCAT_STRIDE = 5

PRIME_ACKNOWLEDGMENT = (
    "Estimates solved with the PRIME estimator (well-robotics/PRIME); "
    "thanks to the PRIME authors for their excellent work."
)


def resolve_result_dir(clip: str, result_dir: str | None = None) -> Path:
    """A saved solution bundle: explicit dir or the shipped reference."""
    if result_dir is not None:
        return resolve_inside_root(result_dir)
    return resolve_inside_root(f"data/clips/{clip}/reference_solution")


def build_sequence(
    clip: str,
    *,
    result_dir: str | None = None,
    style_config: str = DEFAULT_STYLE,
):
    if clip not in component_specs():
        raise ValueError(f"unknown clip {clip!r}; expected run1 or run2")
    result = resolve_result_dir(clip, result_dir)
    builder = MotionForceSequenceBuilder(load_visualization_style(style_config))
    sequence = builder.build(
        result_dir=str(result.relative_to(project_root())),
        config=str((result / "request_config.xml").relative_to(project_root())),
        profile_id="g1",
        run_id=f"{clip}_calibrated",
        gt_clip=load_gt_clip(clip),
    )
    if sequence.number_of_states != HORIZON_STATES:
        raise RuntimeError("sequence state count mismatch")
    if sequence.number_of_intervals != HORIZON_TRANSITIONS:
        raise RuntimeError("sequence interval count mismatch")
    return sequence, result


def render_clip(
    clip: str,
    *,
    result_dir: str | None = None,
    output_root: str = "out/render",
    style_config: str = DEFAULT_STYLE,
    meshcat_stride: int = MESHCAT_STRIDE,
) -> dict:
    started = time.perf_counter()
    sequence, result = build_sequence(
        clip, result_dir=result_dir, style_config=style_config
    )
    output = resolve_inside_root(
        f"{output_root}/{clip}", must_exist=False
    )
    output.mkdir(parents=True, exist_ok=True)
    relative = lambda path: str(Path(path).relative_to(project_root()))
    manifest_path, sequence_path = write_sequence_artifacts(
        sequence, relative(output)
    )
    metrics_path = write_frame_metrics(
        sequence, relative(output / "frame_metrics_50hz.csv")
    )
    report = {
        "schema": "g1cal_render_report_v1",
        "clip": clip,
        "source_result": relative(result),
        "statement": CALIBRATION_STATEMENT,
        "acknowledgment": PRIME_ACKNOWLEDGMENT,
        "states": HORIZON_STATES,
        "transitions": HORIZON_TRANSITIONS,
        "sequence_manifest": relative(manifest_path),
        "sequence": relative(sequence_path),
        "frame_metrics": relative(metrics_path),
        "artifacts": {},
    }
    truth = np.loadtxt(
        resolve_inside_root(component_specs()[clip].upper_truth),
        delimiter=",", ndmin=2,
    )
    loss = trajectory_loss_arrays(sequence.physical_states, truth)
    report["se3_log_loss"] = loss.value
    report["branch_margin_rad"] = loss.branch_margin_rad
    before = time.perf_counter()
    meshcat_path = record_meshcat_html(
        sequence,
        relative(output / "meshcat_interactive.html"),
        stride=meshcat_stride,
    )
    indices = sampled_transition_indices(
        HORIZON_TRANSITIONS, stride=meshcat_stride
    )
    report["artifacts"]["meshcat_html"] = {
        "path": relative(meshcat_path),
        "frames": len(indices),
        "fps": sampled_playback_fps(indices, sequence.dt),
        "default_playback_rate": 0.5,
        "available_playback_rates": [0.5, 1.0],
        "final_transition_included": int(indices[-1]) == (
            HORIZON_TRANSITIONS - 1
        ),
        "seconds": time.perf_counter() - before,
    }
    report["wall_seconds"] = time.perf_counter() - started
    atomic_write_json(output / "render_report.json", report)
    return report


def replay_clip(
    clip: str,
    *,
    result_dir: str | None = None,
    style_config: str = DEFAULT_STYLE,
    loop: bool = True,
    realtime_factor: float = 1.0,
) -> None:
    """Interactive MuJoCo viewer replay (kinematic; qpos/qvel + mj_forward)."""
    from .visualization.mujoco_renderer import replay_interactive

    sequence, _ = build_sequence(
        clip, result_dir=result_dir, style_config=style_config
    )
    replay_interactive(
        sequence, loop=loop, realtime_factor=realtime_factor
    )
