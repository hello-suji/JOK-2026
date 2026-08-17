from __future__ import annotations

import io
import json
import shlex
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import docker

from src.contracts.instance_views import PrePatchInstanceView
from src.evaluator.dynamic_slice_cc import (
    BACKEND_NAME,
    CC_INSTRUMENTATION_ERROR,
    CC_TIMEOUT,
    DynamicSliceCCResult,
    dynamic_slice_result_from_dict,
)
from src.executor.alignment_runner import _cleanup_container, _exec_with_timeout
from src.utils.artifact_hash import sha256_text
from src.utils.file_io import write_json_atomic


MAX_CAPTURE_CHARS = 12000
CONTAINER_REPO_PATH = Path("/testbed")
CONTAINER_WORKER_ROOT = Path("/tmp/m8_dynamic_slice_worker")
CONTAINER_WORKER_PATH = CONTAINER_WORKER_ROOT / "m8_dynamic_slice_worker.py"
CONTAINER_PATCH_PATH = Path("/tmp/m8_generated_test.patch")
CONTAINER_ARTIFACT_PATH = Path("/tmp/m8_dynamic_slice_cc.json")


@dataclass(frozen=True)
class M8DynamicSliceRequest:
    """Pre-patch-only input for M8 Checked Coverage tracing.

    The request intentionally has no golden-patch, post-patch, final-harness
    outcome, or Patch Hit Rate fields. CC is computed from base commit plus the
    generated test patch only.
    """

    instance: PrePatchInstanceView
    test_nodeid: str
    output_path: Path
    generated_test_patch: str
    generated_patch_path: Optional[Path] = None
    generated_patch_sha256: Optional[str] = None
    timeout_seconds: int = 600


class M8DynamicSliceRunner:
    """Run M8 dynamic-slice Checked Coverage inside the benchmark container."""

    def __init__(self, *, docker_client: Any | None = None) -> None:
        self._client = docker_client if docker_client is not None else docker.from_env()

    def run(self, request: M8DynamicSliceRequest) -> DynamicSliceCCResult:
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = _base_diagnostics(request)
        if not request.generated_test_patch.strip():
            result = _instrumentation_error(
                "generated_test_patch_missing",
                diagnostics,
            )
            write_json_atomic(result.to_dict(), output_path)
            return result
        if not request.test_nodeid or "::" not in request.test_nodeid:
            result = _instrumentation_error(
                "pytest_collection_failed",
                {**diagnostics, "test_nodeid_valid": False},
            )
            write_json_atomic(result.to_dict(), output_path)
            return result

        try:
            image_name = self._ensure_instance_image(request.instance)
        except docker.errors.ImageNotFound as exc:
            result = _instrumentation_error(
                "container_image_unavailable",
                {**diagnostics, "error": str(exc)},
            )
            write_json_atomic(result.to_dict(), output_path)
            return result
        except Exception as exc:
            result = _instrumentation_error(
                "benchmark_environment_unavailable",
                {**diagnostics, "error": str(exc)},
            )
            write_json_atomic(result.to_dict(), output_path)
            return result

        diagnostics["container_image"] = image_name
        container = None
        command = _build_container_command(request)
        diagnostics["pytest_command"] = command
        try:
            container_name = f"sweb.m8cc.{request.instance.instance_id}.{uuid.uuid4().hex[:8]}"
            container = self._client.containers.create(
                image_name,
                name=container_name,
                detach=True,
                tty=True,
            )
            container.start()
            _copy_text_to_container(container, request.generated_test_patch, CONTAINER_PATCH_PATH)
            _copy_worker_package(container)
            stdout, timed_out, elapsed = _exec_with_timeout(
                container,
                command,
                request.timeout_seconds,
            )
            container_python_version = _container_python_version_from_stdout(stdout)
            if container_python_version:
                diagnostics["container_python_version"] = container_python_version
            diagnostics["captured_stdout"] = _tail(stdout)
            diagnostics["captured_stderr"] = ""
            diagnostics["execution_time_seconds"] = elapsed
            if timed_out:
                result = DynamicSliceCCResult(
                    backend=BACKEND_NAME,
                    status=CC_TIMEOUT,
                    diagnostics={
                        **diagnostics,
                        "reason": "dynamic_slice_timeout",
                        "timeout_seconds": request.timeout_seconds,
                    },
                )
                write_json_atomic(result.to_dict(), output_path)
                return result
            if "M8_PATCH_APPLY_FAILED" in stdout:
                result = _instrumentation_error(
                    "generated_test_patch_apply_failed",
                    {**diagnostics, "captured_stdout": _tail(stdout)},
                )
                write_json_atomic(result.to_dict(), output_path)
                return result
            artifact_payload = _read_json_from_container(container, CONTAINER_ARTIFACT_PATH)
            if artifact_payload is None:
                result = _instrumentation_error(
                    "dynamic_slice_artifact_missing",
                    diagnostics,
                )
                write_json_atomic(result.to_dict(), output_path)
                return result
            try:
                result = dynamic_slice_result_from_dict(artifact_payload)
            except (TypeError, ValueError) as exc:
                result = _instrumentation_error(
                    "malformed_dynamic_slice_artifact",
                    {**diagnostics, "error": str(exc)},
                )
                write_json_atomic(result.to_dict(), output_path)
                return result
            result.diagnostics = {
                **diagnostics,
                **dict(result.diagnostics),
                "patch_applied": True,
                "patch_application_status": "applied",
                "worker_compatibility_mode": "standalone_py36",
            }
            write_json_atomic(result.to_dict(), output_path)
            return result
        except Exception as exc:
            result = _instrumentation_error(
                "benchmark_environment_unavailable",
                {**diagnostics, "error": str(exc)},
            )
            write_json_atomic(result.to_dict(), output_path)
            return result
        finally:
            _cleanup_container(self._client, container)

    def _ensure_instance_image(self, instance: PrePatchInstanceView) -> str:
        from tddbench.harness.test_spec import make_test_spec

        tdd_image_raw = instance.to_tdd_image_raw()
        spec = make_test_spec(tdd_image_raw)
        image_name = spec.instance_image_key
        try:
            self._client.images.get(image_name)
            return image_name
        except docker.errors.ImageNotFound:
            from tddbench.harness.docker_build import build_instance_images

            successful, failed = build_instance_images(
                self._client,
                [tdd_image_raw],
                force_rebuild=False,
                max_workers=1,
            )
            if failed or not successful:
                raise RuntimeError(f"Docker image build failed: {image_name}")
            self._client.images.get(image_name)
            return image_name


def make_m8_dynamic_slice_request(
    *,
    instance: PrePatchInstanceView,
    generated_test: Mapping[str, Any],
    generated_patch_path: Path,
    test_nodeid: str,
    output_path: Path,
    timeout_seconds: int = 600,
) -> M8DynamicSliceRequest:
    patch_text, provenance_path = _canonical_generated_patch(
        generated_test=generated_test,
        generated_patch_path=generated_patch_path,
    )
    patch_sha = sha256_text(patch_text) if patch_text else None
    return M8DynamicSliceRequest(
        instance=instance,
        test_nodeid=test_nodeid,
        output_path=output_path,
        generated_test_patch=patch_text,
        generated_patch_path=provenance_path,
        generated_patch_sha256=patch_sha,
        timeout_seconds=timeout_seconds,
    )


def _canonical_generated_patch(
    *,
    generated_test: Mapping[str, Any],
    generated_patch_path: Path,
) -> tuple[str, Optional[Path]]:
    if generated_patch_path.exists():
        return generated_patch_path.read_text(encoding="utf-8"), generated_patch_path
    patch_text = generated_test.get("test_patch")
    if isinstance(patch_text, str) and patch_text.strip():
        return patch_text, None
    alt_path = generated_test.get("generated_patch_path")
    if isinstance(alt_path, str) and alt_path.strip():
        path = Path(alt_path)
        if path.exists():
            return path.read_text(encoding="utf-8"), path
    return "", None


def _build_container_command(request: M8DynamicSliceRequest) -> str:
    quoted_nodeid = shlex.quote(request.test_nodeid)
    script = "\n".join(
        [
            "set -euxo pipefail",
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"cd {CONTAINER_REPO_PATH}",
            f"git config --global --add safe.directory {CONTAINER_REPO_PATH}",
            f"git reset --hard {shlex.quote(request.instance.base_commit)}",
            "git clean -fd",
            f"git apply -v {CONTAINER_PATCH_PATH} || {{ echo M8_PATCH_APPLY_FAILED; exit 17; }}",
            "printf 'M8_CONTAINER_PYTHON_VERSION='",
            "python -c 'import sys; print(sys.version.split()[0])'",
            (
                f"python {CONTAINER_WORKER_PATH} "
                f"--repo {CONTAINER_REPO_PATH} "
                f"--nodeid {quoted_nodeid} "
                f"--output {CONTAINER_ARTIFACT_PATH}"
            ),
        ]
    )
    return f"/bin/bash -lc {shlex.quote(script)}"


def _base_diagnostics(request: M8DynamicSliceRequest) -> dict[str, Any]:
    return {
        "backend": BACKEND_NAME,
        "instance_id": request.instance.instance_id,
        "execution_environment": "benchmark_instance_container",
        "testbed_path": str(CONTAINER_REPO_PATH),
        "test_nodeid": request.test_nodeid,
        "generated_test_patch_provenance": (
            str(request.generated_patch_path) if request.generated_patch_path else "generated_test.test_patch"
        ),
        "generated_patch_sha256": (
            request.generated_patch_sha256
            or (sha256_text(request.generated_test_patch) if request.generated_test_patch else None)
        ),
        "patch_applied": None,
        "patch_application_status": None,
        "container_python_version": None,
        "worker_compatibility_mode": "standalone_py36",
        "collection_status": None,
        "execution_status": None,
    }


def _instrumentation_error(reason: str, diagnostics: Mapping[str, Any]) -> DynamicSliceCCResult:
    patch_applied = diagnostics.get("patch_applied")
    patch_application_status = diagnostics.get("patch_application_status")
    if reason == "generated_test_patch_apply_failed":
        patch_applied = False
        patch_application_status = "failed"
    return DynamicSliceCCResult(
        backend=BACKEND_NAME,
        status=CC_INSTRUMENTATION_ERROR,
        diagnostics={
            **dict(diagnostics),
            "reason": reason,
            "patch_applied": patch_applied,
            "patch_application_status": patch_application_status,
        },
    )


def _copy_text_to_container(container: Any, text: str, dst: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, dir="/tmp") as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        _copy_file_to_container(container, tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


def _copy_worker_package(container: Any) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    worker_path = repo_root / "src" / "executor" / "m8_dynamic_slice_worker_py36.py"
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(worker_path, arcname=str(CONTAINER_WORKER_PATH))
    tar_stream.seek(0)
    container.exec_run(f"mkdir -p {CONTAINER_WORKER_ROOT}")
    container.put_archive("/", tar_stream.read())


def _copy_file_to_container(container: Any, src: Path, dst: Path) -> None:
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(str(src), arcname=dst.name)
    tar_stream.seek(0)
    container.exec_run(f"mkdir -p {dst.parent}")
    container.put_archive(str(dst.parent), tar_stream.read())


def _read_json_from_container(container: Any, src: Path) -> Optional[dict[str, Any]]:
    try:
        stream, _stat = container.get_archive(str(src))
    except Exception:
        return None
    data = b"".join(stream)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tar:
        members = tar.getmembers()
        if not members:
            return None
        extracted = tar.extractfile(members[0])
        if extracted is None:
            return None
        payload = json.loads(extracted.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else None


def _tail(value: str, *, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _container_python_version_from_stdout(stdout: str) -> Optional[str]:
    for line in str(stdout or "").splitlines():
        if line.startswith("M8_CONTAINER_PYTHON_VERSION="):
            return line.split("=", 1)[1].strip() or None
    return None
