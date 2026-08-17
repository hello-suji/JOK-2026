from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.artifact_hash import sha256_text
from src.utils.file_io import write_json_atomic


@dataclass
class HarnessExecutionResult:
    instance_id: str
    benchmark_root: str
    predictions_path: str
    run_id: str
    harness_command: List[str]
    harness_returncode: int
    harness_stdout: str
    harness_stderr: str
    internal_failure: bool = False
    internal_failure_reason: Optional[str] = None
    internal_failure_artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ReproductionTestRunner:
    def __init__(
        self,
        benchmark_root: str = "benchmark/TDD-Bench-Verified",
        max_workers: int = 1,
    ) -> None:
        self.benchmark_root = Path(benchmark_root).resolve()
        self.max_workers = max_workers

    def run(
        self,
        instance_id: str,
        generated_test_json_path: str,
        run_id: Optional[str] = None,
    ) -> HarnessExecutionResult:
        generated_path = Path(generated_test_json_path).resolve()
        if not generated_path.exists():
            raise FileNotFoundError(f"generated_test.json 파일이 없습니다: {generated_path}")

        patch_path = generated_path.with_suffix(".patch").resolve()
        if not patch_path.exists():
            raise FileNotFoundError(f"generated_test.patch 파일이 없습니다: {patch_path}")

        if not self.benchmark_root.exists():
            raise FileNotFoundError(f"TDD-Bench-Verified 경로가 없습니다: {self.benchmark_root}")

        patch_text = patch_path.read_text(encoding="utf-8")
        patch_sha256 = sha256_text(patch_text)

        predictions_path = generated_path.with_name("predictions.json")
        predictions = [
            {
                "instance_id": instance_id,
                "model_patch": patch_text,
                "patch_sha256": patch_sha256,
            }
        ]
        write_json_atomic(predictions, predictions_path)

        run_id = run_id or f"debug-{instance_id}"

        command = [
            "python",
            "-m",
            "tddbench.harness.run_evaluation",
            "--dataset_name",
            "TDD_Bench.json",
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(self.max_workers),
            "--instance_ids",
            instance_id,
            "--run_id",
            run_id,
        ]

        try:
            result = subprocess.run(
                command,
                cwd=str(self.benchmark_root),
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            return HarnessExecutionResult(
                instance_id=instance_id,
                benchmark_root=str(self.benchmark_root),
                predictions_path=str(predictions_path),
                run_id=run_id,
                harness_command=command,
                harness_returncode=-1,
                harness_stdout="",
                harness_stderr="Harness timed out after 1800 seconds",
                internal_failure=True,
                internal_failure_reason="Harness timed out after 1800 seconds",
            )
        except Exception as e:
            return HarnessExecutionResult(
                instance_id=instance_id,
                benchmark_root=str(self.benchmark_root),
                predictions_path=str(predictions_path),
                run_id=run_id,
                harness_command=command,
                harness_returncode=-1,
                harness_stdout="",
                harness_stderr=f"subprocess error: {e}",
                internal_failure=True,
                internal_failure_reason=f"subprocess error: {e}",
            )

        internal_failure, reason, artifacts = self._detect_internal_failure(
            instance_id=instance_id,
            run_id=run_id,
            stdout=result.stdout,
            stderr=result.stderr,
        )

        return HarnessExecutionResult(
            instance_id=instance_id,
            benchmark_root=str(self.benchmark_root),
            predictions_path=str(predictions_path),
            run_id=run_id,
            harness_command=command,
            harness_returncode=result.returncode,
            harness_stdout=result.stdout,
            harness_stderr=result.stderr,
            internal_failure=internal_failure,
            internal_failure_reason=reason,
            internal_failure_artifacts=artifacts,
        )

    def _detect_internal_failure(
        self,
        *,
        instance_id: str,
        run_id: str,
        stdout: str,
        stderr: str,
    ) -> tuple[bool, Optional[str], List[str]]:
        """Detect harness-internal errors even when the wrapper exits zero."""
        combined = f"{stdout}\n{stderr}"
        errors = re.findall(r"Instances with errors:\s*(\d+)", combined, re.IGNORECASE)
        if any(int(value) > 0 for value in errors):
            return True, "benchmark harness reported Instances with errors", []
        marker_patterns = (
            r"Error in evaluating model for\s+" + re.escape(instance_id),
            r"Traceback \(most recent call last\)",
            r"AttributeError:.*get_all_classes",
            r"ModuleNotFoundError:",
        )
        for pattern in marker_patterns:
            if re.search(pattern, combined, re.IGNORECASE):
                return True, f"benchmark harness internal failure matched {pattern}", []

        artifacts: List[str] = []
        log_root = self.benchmark_root / "logs" / "run_evaluation" / run_id
        if log_root.exists():
            for log_path in log_root.rglob("run_instance.log"):
                try:
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if (
                    "Traceback (most recent call last)" in log_text
                    or "get_all_classes" in log_text
                    or "Error in evaluating model" in log_text
                ):
                    artifacts.append(str(log_path))
            if artifacts:
                return True, "benchmark per-instance log contains an internal error", artifacts
        return False, None, artifacts

    def save(self, result: HarnessExecutionResult, output_path: str) -> None:
        write_json_atomic(result.to_dict(), output_path)
