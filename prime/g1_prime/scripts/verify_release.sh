#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

full=0
if [[ "${1:-}" == "--full" ]]; then
  full=1
elif [[ $# -ne 0 ]]; then
  echo "usage: scripts/verify_release.sh [--full]" >&2
  exit 2
fi

command -v python >/dev/null
command -v cmake >/dev/null

./scripts/build.sh
ctest --test-dir build/overlay --output-on-failure
python -m pip install -e .
pytest -q tests/python
python scripts/publish_pages.py \
  --work-root out/verify_pages_build \
  --site-root out/verify_pages >/dev/null

python - <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

from g1cal.calibration import calibrated_theta
from g1cal.horizon_solver import validate_warm_start_dir
from g1cal.profiles import load_model_profile
from g1cal.rendering import build_sequence

root = Path.cwd()
profile = load_model_profile("g1")
assert profile.scene_mjcf_path is not None
scene = ET.parse(profile.scene_mjcf_path).getroot()
ground = scene.find("asset/texture[@name='groundplane']")
assert ground is not None
assert ground.attrib["rgb1"] == "0.2 0.3 0.4"
assert ground.attrib["rgb2"] == "0.1 0.2 0.3"
assert ground.attrib["markrgb"] == "0.8 0.8 0.8"
assert calibrated_theta().shape == (17,)
for clip in ("run1", "run2"):
    clip_root = root / "data/clips" / clip
    expected = {"q_sense.csv": 501, "v_sense.csv": 501,
                "tau_sense.csv": 500, "upper_truth_h501.csv": 501}
    for name, rows in expected.items():
        assert np.loadtxt(clip_root / name, delimiter=",", ndmin=2).shape[0] == rows
    validate_warm_start_dir(
        f"data/clips/{clip}/reference_solution", expected_profile_id="g1"
    )
    reference = clip_root / "reference_solution"
    assert not (reference / "contact_candidate_diagnostics.csv").exists()
    assert not (reference / "contact_corner_diagnostics.csv").exists()
    sequence, _ = build_sequence(clip)
    assert sequence.number_of_states == 501
    assert sequence.number_of_intervals == 500
summary = json.loads((root / "data/calibrated/calibration_summary.json").read_text())
assert summary["schema"] == "g1cal_calibration_summary_v1"
landing_text = (root / "docs/index.html").read_text()
assert "<title>Calibrated Contact-Aware Full-Information Estimation</title>" in landing_text
assert "The light-gray robot is the calibrated estimate" in landing_text
assert "Blue/yellow arrows: estimated left/right contact forces" in landing_text
assert "pink: ground-truth foot forces" in landing_text
assert "green point: estimated CoP" in landing_text
assert "Kang et al.</a>—" in landing_text
assert "(BSD-3-Clause)" not in landing_text
assert "well-robotics), BSD-3" not in landing_text
assert "<footer>" not in landing_text
assert landing_text.count("<iframe") == 1
assert 'id="viewer-frame"' in landing_text
assert 'src="media/run1_calibrated.html"' in landing_text
assert landing_text.count("Load 0.5x") == 2
assert landing_text.count("Load 1x") == 2
assert 'mixer.addEventListener("finished", advance)' in landing_text
assert "loadSegment((segmentIndex + 1) % segments.length, playbackRate)" in landing_text
for clip in ("run1", "run2"):
    interactive = root / "out/verify_pages/media" / f"{clip}_calibrated.html"
    assert 1_000_000 < interactive.stat().st_size < 100_000_000
    interactive_text = interactive.read_text()
    assert 'id="g1cal-follow-camera-v1"' in interactive_text
    assert 'id="g1cal-playback-rate-v1"' in interactive_text
    assert "setRate(requested === 1.0 ? 1.0 : 0.5)" in interactive_text
assert not list((root / "docs").rglob("*.gif"))
assert not (root / "docs/media").exists()
assert not any(
    path.stat().st_size >= 100_000_000
    for path in root.rglob("*")
    if path.is_file() and not any(part in {".git", "build", "out"} for part in path.parts)
)
source_paths = subprocess.check_output(
    ["git", "ls-files", "-co", "--exclude-standard", "-z"]
).decode().split("\0")
oversized = [
    (relative, (root / relative).stat().st_size)
    for relative in source_paths if relative and (root / relative).is_file()
    and (root / relative).stat().st_size >= 50 * 1024 * 1024
]
assert not oversized, f"regular-Git source files exceed 50 MiB: {oversized}"
PY

g1cal --help >/dev/null
g1cal calibrate --help >/dev/null
term_a='over''fit'
term_b='held''.?''out'
old_namespace='g1''_prime'
scan_args=(--exclude-dir=third_party --exclude-dir=build --exclude-dir=out \
  --exclude-dir=.git --exclude-dir=__pycache__ --exclude='*.pyc' \
  --exclude='*.so')
if grep -riE "${scan_args[@]}" "$term_a|$term_b" .; then
  echo "public terminology scan failed" >&2
  exit 1
fi
if grep -ri "${scan_args[@]}" "$old_namespace" .; then
  echo "old namespace scan failed" >&2
  exit 1
fi
grep -q "built on top of PRIME's excellent estimator" NOTICE.md
grep -q "PRIME's excellent estimator" docs/index.html
git check-attr filter -- models/g1/mjcf/assets/pelvis_contour_link.STL | grep -q ': lfs$'

if [[ $full -eq 1 ]]; then
  g1cal solve --clip run1 --covariance data/calibrated/precision.csv \
    --out out/release_full
  g1cal solve --clip run2 --covariance data/calibrated/precision.csv \
    --out out/release_full
  g1cal calibrate --optimizer sqp-bfgs --max-iterations 1 \
    --out out/release_full
  g1cal calibrate --optimizer frank-wolfe-sdp --max-iterations 1 \
    --out out/release_full
fi

permission_status="$(python - <<'PY'
import json
from pathlib import Path
path = Path("data/clips/PUBLICATION_STATUS.json")
print(json.loads(path.read_text()).get("status", "missing") if path.is_file() else "missing")
PY
)"
if [[ "$permission_status" != "authorized" ]]; then
  echo "technical verification passed; motion-data publication permission is not authorized" >&2
  exit 3
fi

echo "release verification passed"
