"""Load, or locally compile, generated CasADi kinematics functions."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from casadi import external


@dataclass(frozen=True)
class CodegenFunctions:
    foot_velocity: object
    foot_position: object


class CodegenLibraryLoader:
    def __init__(self, source_dir: str | Path, cache_dir: str | Path | None = None):
        self.source_dir = Path(source_dir)
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "legbical")

    def load(self) -> CodegenFunctions:
        return CodegenFunctions(
            foot_velocity=external("yv_and_J", str(self._library("yv_and_J_codegen"))),
            foot_position=external("pf_and_J", str(self._library("pf_and_J_codegen"))),
        )

    def _library(self, stem: str) -> Path:
        suffix = self._suffix()
        packaged = self.source_dir / f"lib{stem}{suffix}"
        if packaged.exists():
            return packaged
        output = self.cache_dir / f"lib{stem}{suffix}"
        if output.exists():
            return output
        source = self.source_dir / f"{stem}.c"
        if not source.exists():
            raise FileNotFoundError(source)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        compiler = shutil.which(os.environ.get("CC", "cc"))
        if compiler is None:
            raise RuntimeError("a C compiler is required to build kinematics code")
        with tempfile.TemporaryDirectory(dir=self.cache_dir) as temporary:
            candidate = Path(temporary) / output.name
            command = [compiler, "-O3", "-fPIC"]
            if platform.system() == "Darwin":
                command += ["-dynamiclib", str(source), "-o", str(candidate)]
            elif platform.system() == "Linux":
                command += ["-shared", str(source), "-lm", "-o", str(candidate)]
            else:
                raise RuntimeError("automatic codegen compilation supports Linux and macOS")
            subprocess.run(command, check=True, capture_output=True, text=True)
            os.replace(candidate, output)
        return output

    @staticmethod
    def _suffix() -> str:
        if platform.system() == "Darwin":
            return ".dylib"
        if platform.system() == "Windows":
            return ".dll"
        return ".so"
