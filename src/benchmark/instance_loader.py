from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.instance_views import (
    M8EvaluationInstanceView,
    PrePatchInstanceView,
    make_m8_evaluation_view,
    make_pre_patch_view,
)


@dataclass
class BenchmarkInstance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str = ""
    test_patch: str = ""
    created_at: str = ""
    version: str = ""
    environment_setup_commit: str = ""
    difficulty: str = ""
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_pre_patch_view(self) -> PrePatchInstanceView:
        return make_pre_patch_view(self)

    def to_m8_evaluation_view(self) -> M8EvaluationInstanceView:
        return make_m8_evaluation_view(self)


class TDDInstanceLoader:
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = Path(dataset_path) if dataset_path else self._find_dataset_path()

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"TDD_Bench.json 파일을 찾지 못했습니다: {self.dataset_path}\n"
                f"benchmark/TDD-Bench-Verified 안에서 dataset_preparation.py를 먼저 실행했는지 확인하세요."
            )

    def _find_dataset_path(self) -> Path:
        """
        프로젝트 루트 기준으로 TDD_Bench.json 위치를 자동 탐색한다.
        """
        project_root = Path(__file__).resolve().parents[2]

        candidates = [
            project_root / "benchmark" / "TDD-Bench-Verified" / "TDD_Bench.json",
            project_root / "benchmark" / "TDD-Bench-Verified" / "data" / "TDD_Bench.json",
            project_root / "data" / "TDD_Bench.json",
        ]

        for path in candidates:
            if path.exists():
                return path

        # 못 찾았을 때 에러 메시지용 기본 경로
        return candidates[0]

    @lru_cache(maxsize=1)
    def load_all(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"TDD_Bench.json 파싱 실패: {e}") from e

        if isinstance(data, dict):
            if "instances" in data and isinstance(data["instances"], list):
                return data["instances"]
            raise ValueError("지원하지 않는 JSON 구조입니다. list 또는 {'instances': [...]} 형식을 기대합니다.")

        if not isinstance(data, list):
            raise ValueError("TDD_Bench.json은 list 형식이어야 합니다.")

        return data

    def get_instance(self, instance_id: str) -> BenchmarkInstance:
        for item in self.load_all():
            if item.get("instance_id") == instance_id:
                return self._to_instance(item)

        raise KeyError(f"instance_id='{instance_id}' 를 찾지 못했습니다.")

    def get_by_index(self, index: int) -> BenchmarkInstance:
        data = self.load_all()

        if index < 0 or index >= len(data):
            raise IndexError(f"index={index} 는 범위를 벗어났습니다. 전체 개수: {len(data)}")

        return self._to_instance(data[index])

    def list_instance_ids(self, limit: Optional[int] = None) -> List[str]:
        ids = [item.get("instance_id", "") for item in self.load_all()]
        return ids if limit is None else ids[:limit]

    def _to_instance(self, item: Dict[str, Any]) -> BenchmarkInstance:
        return BenchmarkInstance(
            instance_id=item.get("instance_id", ""),
            repo=item.get("repo", ""),
            base_commit=item.get("base_commit", ""),
            problem_statement=item.get("problem_statement", ""),
            patch=item.get("patch", ""),
            test_patch=item.get("test_patch", ""),
            created_at=item.get("created_at", ""),
            version=item.get("version", ""),
            environment_setup_commit=item.get("environment_setup_commit", ""),
            difficulty=item.get("difficulty", ""),
            raw=item,
        )


if __name__ == "__main__":
    loader = TDDInstanceLoader()

    print(f"dataset_path = {loader.dataset_path}")
    print(f"num_instances = {len(loader.load_all())}")

    sample = loader.get_by_index(0)
    print("\n[sample instance]")
    print("instance_id:", sample.instance_id)
    print("repo:", sample.repo)
    print("base_commit:", sample.base_commit)
    print("problem_statement preview:")
    print(sample.problem_statement[:300])
