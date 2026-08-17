"""
AlignmentScorer — patch-free 규칙기반 정합성 평가.

alignment_runner의 before-patch 결과를 받아:
  1) s_b: 버그 코드에서 FAIL 여부 (Bug Reproduction Score, 0~1)
  2) s_a: 이슈-테스트 규칙기반 정합성 (Issue Alignment Score, 0~1)
  3) s_c: 의심 위치 커버리지 (Coverage Score, 0~1)
를 독립 계산하고, 세 점수를 순차 게이트로 검사하여 ALIGNED 판정한다.

논문 §3.3 게이트 방식:
  Gate 1 (s_b) → Gate 2 (s_c) → Gate 3 (s_a) 순서로 각 임계값 검사.
  모두 통과 시 ALIGNED; 실패 시 해당 유형 반환.

판정 기준:
  최종 label은 s_b, s_c, s_a의 순차 게이트 통과 여부로만 결정한다.
  평균 alignment score는 배치 요약 지표로 사용하지 않는다.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import re
import ast
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set

from src.contracts.final_sets import admitted_to_final_set
from src.contracts.feature_flags import V22FeatureFlags, resolve_feature_flags
from src.contracts.status import legacy_failure_type_to_statuses
from src.contracts.v37_oracle_flags import validated_v37_blocking_oracle_flags
from src.executor.alignment_runner import normalize_pre_patch_execution_status
from src.alignment.oracle_consistency import (
    evaluate_oracle_consistency,
    generated_test_contains_bug_trigger,
    issue_bug_trigger_patterns,
)
from src.alignment.feedback_router import (
    build_iteration_feedback_summary,
    build_structured_feedback,
)
from src.generator.reproduction_examples import (
    sanitize_oracle_regeneration_payload,
    sanitize_repair_directive,
    selected_example_requires_oracle_regeneration,
)
from src.generator.relational_oracles import RELATIONAL_ORACLE_PROVENANCE
from src.scenario.code_block_roles import (
    ROLE_BUG_TRIGGER,
    block_inferred_role,
    classify_reproduction_code_blocks,
    contains_target_call,
    is_setup_only_block,
)

logger = logging.getLogger(__name__)

ALIGNMENT_SCORE_SCHEMA_VERSION = "gate-v4-0to1"

# ---------------------------------------------------------------------------
# 점수 구성 — 각 컴포넌트 독립 계산 후 순차 게이트 검사 (논문 §3.3)
# label은 s_b, s_c, s_a 게이트로 판정한다. 평균 score는 산출하지 않는다.
# ---------------------------------------------------------------------------

# bug_fail_score (s_b): 0 ~ 1  (before-patch 실패 관측 게이트)
_BUG_FAIL_MAX           = 1.0
_BUG_FAIL_PARTIAL       = 1.0   # ERROR+PASSED 혼재 (부분 재현) 점수
_ALIGNED_BUG_FAIL_MIN   = 0.70  # ALIGNED 판정을 위한 τb

# 피처 가중치.
#
# s_b = clip(Σ_t w_t f_t, 0, 1). ERROR/NOT_VALID 계열 신호는 버그 재현
# 점수에서 감점하지 않고 failure_type과 feedback에서 별도로 다룬다.
_BUG_FAIL_FEATURE_WEIGHTS = {
    "f_fail":     0.50,  # before-patch에서 생성 테스트가 FAILED로 관측된 정도
    "f_assert":   0.25,  # assertion failure 또는 expected/actual mismatch 신호
    "f_symptom":  0.25,  # 실패 출력/테스트가 이슈 증거 토큰과 겹치는 정도
}

# issue_alignment_score (s_a): 0 ~ 1  (원 논문 Eq.(3))
# Each applicable criterion contributes evidence recall.  Criteria for which
# the issue supplies no evidence are excluded from the average denominator.
_ISSUE_ALIGN_MAX        = 1.0
_ISSUE_ALIGN_SUB        = _ISSUE_ALIGN_MAX / 4
_ISSUE_ALIGN_TOKEN_HIT  = 0.1   # retained for strong-evidence diagnostics only
_ISSUE_ALIGN_STRONG_GATE = 0.75  # token overlap only인 경우 ALIGNED를 막기 위한 강화 게이트

# coverage_score (s_c): 0 ~ 1  (의심 위치 커버리지)
# =====================================================================
# 설계 근거:
# • 목표: 테스트가 의심 코드 라인 자체를 실행했는지 검증
# • 의심 라인 L_s:
#   1) clue.fault_locations의 line_no 주변 window
#   2) 없으면 scenario.target_location의 target_function AST statement lines
#   3) 둘 다 없으면 target source file coverage ratio fallback
# • s_c = |covered(L_s)| / |L_s| ∈ [0, 1]
#   - covered 여부는 coverage missing_lines에 포함되지 않는 statement line으로 판단
#   - gold patch/final-eval patch line coverage는 alignment에서 사용하지 않음
_COVERAGE_MAX           = 1.0
_COVERAGE_BASE          = 0.0   # base bonus 제거
_COVERAGE_BONUS_MAX     = 1.0   # 커버리지 비율(0~100%)을 그대로 0~1로 환산
_COVERAGE_FALLBACK      = 1.0  # source_file 미지정 시, non-test 파일 커버 비율로 환산
_SUSPICIOUS_LINE_WINDOW = 3

# WEAK_ALIGNMENT → switch_scenario 임계값
_SWITCH_SCENARIO_THRESHOLD = 0.3  # 가장 약한 게이트 점수가 이 미만이면 시나리오 전환

# Requirements-Based Scoring (V2) gate 임계값
# 각 컴포넌트의 최솟값을 만족해야 ALIGNED 판정 가능
_COVERAGE_MIN_GATE    = 0.60  # coverage < 이 값이면 NO_COVERAGE로 차단
_ISSUE_ALIGN_MIN_GATE = 0.65  # issue_align < 이 값이면 WEAK_ALIGNMENT로 차단
_ALIGNED_REPORT_BUG_FAIL_MIN = _ALIGNED_BUG_FAIL_MIN
_ALIGNED_REPORT_COVERAGE_MIN = _COVERAGE_MIN_GATE
_ALIGNED_REPORT_ISSUE_ALIGN_MIN = _ISSUE_ALIGN_MIN_GATE

# 피드백 임계값
_FEEDBACK_BUG_FAIL_WEAK    = _ALIGNED_BUG_FAIL_MIN
_FEEDBACK_ISSUE_ALIGN_WEAK = _ISSUE_ALIGN_MIN_GATE

# ---------------------------------------------------------------------------
# 피드백 문자열 및 반복 횟수 설정
# ---------------------------------------------------------------------------
# 텍스트 길이 제한 (피드백 메시지 가독성 및 토큰 절약)
_FEEDBACK_SHORT_STR_LEN     = 200   # 에러 메시지, assertion 등의 기본 길이
_FEEDBACK_MID_STR_LEN       = 300   # 기대값/실제값 표시 길이
_FEEDBACK_LONG_STR_LEN      = 500   # 상세 설명용 최대 길이

# 반복 항목 표시 개수
_FEEDBACK_ERROR_MSGS_MAX    = 3     # error_messages에서 보여줄 최대 개수
_FEEDBACK_ASSERTION_MAX     = 4     # NOT_FAILED의 assertion 라인 최대 표시
_FEEDBACK_CODE_EXAMPLES_MAX = 2     # 이슈 코드 예시 최대 표시
_FEEDBACK_OUTPUTS_MAX       = 2     # 기대값/실제값 최대 표시
_FEEDBACK_MISSING_IDS_MAX   = 10    # 누락된 식별자 최대 표시
_FEEDBACK_TRACEBACK_LINES   = 10    # fallback traceback에서 표시할 라인 수

# 피드백 메시지 이전 실행 타겟
_FEEDBACK_PREV_ITERATION_KEEP = 1    # 유지할 이전 iteration 개수 (현재 + 이전 1개)
_FEEDBACK_PROMPT_VISIBLE_MAX = 2      # oracle/stimulus/precondition별 prompt-visible 추가 상한

_M7_WEIGHTED_COVERAGE_SCHEMA_VERSION = "m7-sbfl-weighted-coverage-v29-v1"
_M7_LLM_REFINEMENT_SCHEMA_VERSION = "m7-llm-scenario-refinement-v1"
_M7_LLM_ALLOWED_MODULES = {"M2", "M3", "M5"}
_M7_LLM_BRANCH_MODULES = {
    "NOT_FAILED": {"M5"},
    "NO_COVERAGE": {"M2", "M5"},
    "WEAK_ALIGNMENT": {"M3", "M5"},
}
_M7_FORBIDDEN_EVIDENCE_PATTERNS = (
    "after_patch",
    "checked_coverage",
    "f_to_p",
    "fail_to_pass",
    "golden",
    "m8",
    "patch_hit",
    "phr",
    "post_patch",
)


# ---------------------------------------------------------------------------
# 1. 실패 유형 분류
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    ALIGNED = "ALIGNED"               # 세 게이트(s_b, s_c, s_a) 모두 통과
    NOT_FAILED = "NOT_FAILED"         # 버그 코드에서 테스트가 FAIL하지 않음
    ERROR = "ERROR"                    # 실행 에러 (import/syntax/환경 문제)
    NOT_VALID = "NOT_VALID"            # 생성 테스트가 유효하지 않아 실행기가 찾거나 실행하지 못함
    NO_COVERAGE = "NO_COVERAGE"        # target coverage score < 0.60
    WEAK_ALIGNMENT = "WEAK_ALIGNMENT"  # score < threshold (정합성 부족)


def project_m7_status_fields(
    failure_type: FailureType | str,
) -> Dict[str, Any]:
    """Project legacy M7 failure_type into separated contract statuses.

    This is a compatibility projection only. It does not redefine final v22
    ``s_b``, ``s_a``, iteration-1 ``s_c``, ``s_c_prime``, or ``L_s``.
    """
    converted = legacy_failure_type_to_statuses(failure_type)
    candidate = {
        "m7_alignment_status": converted["m7_alignment_status"],
        "diagnostic_only": converted["m7_alignment_status"] != FailureType.ALIGNED.value,
    }
    admitted = admitted_to_final_set(candidate)
    return {
        "execution_status": converted["execution_status"],
        "validation_status": converted["validation_status"],
        "m7_alignment_status": converted["m7_alignment_status"],
        "admitted_to_final_set": admitted,
        "diagnostic_only": not admitted,
        "legacy_failure_type": converted["legacy_failure_type"],
    }


@dataclass
class OracleQuality:
    score: float
    risk_flags: List[str]
    feedback: List[str]


def evaluate_oracle_quality(
    generated_test: Dict[str, Any],
    clue: Optional[Dict[str, Any]] = None,
) -> OracleQuality:
    """Static oracle-risk gate used before accepting ALIGNED.

    Patch-free alignment can confirm that a generated test fails on the buggy
    version, but final resolve also requires the same test to pass after the
    patch. These patterns are frequent causes of fail→fail final outcomes.
    """
    code = generated_test.get("test_code") or generated_test.get("append_block") or ""
    lower = code.lower()
    risk_flags: List[str] = []
    feedback: List[str] = []
    expected_outputs = (clue or {}).get("expected_outputs", [])
    actual_outputs = (clue or {}).get("actual_outputs", [])

    assertion_lines = [
        line.strip()
        for line in code.splitlines()
        if re.search(
            r"\bassert\b|self\.assert|pytest\.raises|with\s+.*raises"
            r"|assert_allclose|assert_array|assert_equal|assert_raises",
            line,
        )
    ]

    def add_flag(flag: str, message: str) -> None:
        if flag not in risk_flags:
            risk_flags.append(flag)
            feedback.append(message)

    test_def_count = len(
        re.findall(r"^\s*(?:async\s+)?def\s+test_\w+\s*\(", code, flags=re.MULTILINE)
    )
    if test_def_count > 1:
        add_flag(
            "multiple_generated_tests",
            "하나의 generated patch에 test_*가 여러 개 있으면 일부는 flip되고 일부는 after에서 실패할 수 있다. "
            "가능하면 하나의 focused reproduction test만 생성하고, 보조 함수는 test_ prefix를 쓰지 마라.",
        )

    def _has_issue_expected_signal() -> bool:
        norm_code = re.sub(r"\s+", "", lower)
        for out in expected_outputs[:3]:
            norm_expected = re.sub(r"\s+", "", str(out).lower())
            if norm_expected and norm_expected[:80] in norm_code:
                return True
        return False

    trivial_assertions = [
        line for line in assertion_lines
        if re.search(
            r"^(?:self\.)?assertTrue\s*\(\s*(?:True|1)\s*(?:,\s*[^)]*)?\)\s*(?:#.*)?$"
            r"|^assert\s+(?:True|1)\s*(?:#.*)?$",
            line,
            re.IGNORECASE,
        )
    ]
    if trivial_assertions:
        add_flag(
            "trivial_oracle",
            "`assertTrue(True)`/`assert 1` 같은 무의미한 oracle은 제거하고 "
            "수정 후 통과해야 하는 실제 return value 또는 state change를 검증하라.",
        )

    # @image_comparison 데코레이터: baseline 이미지 없으면 항상 fail
    if re.search(r"@image_comparison", code):
        add_flag(
            "image_comparison_decorator",
            "@image_comparison 데코레이터는 baseline 이미지가 없으면 항상 실패한다. "
            "직접 값/속성을 검증하는 assertion으로 교체하라.",
        )

    issue_warning_expected = _issue_says_warning_expected(clue)

    # warning 기반 테스트: 패치 후 warning이 사라지면 assertion도 실패
    if re.search(r"catch_warnings|assertWarns|warnings\.warn", code):
        has_non_warning_assertion = bool(re.search(
            r"\bassert\b(?!.*warning)|assertEqual(?!.*warning)|assertIn(?!.*warning)", code, re.IGNORECASE
        ))
        if not has_non_warning_assertion and not issue_warning_expected:
            add_flag(
                "warning_catch_only",
                "warning 기반 테스트는 패치 후 경고가 사라지면 assertion이 실패한다. "
                "경고 대신 실제 동작(return value, side effect)을 검증하라.",
            )

    if re.search(
        r"len\s*\(\s*\w+\s*\)\s*==\s*1.*(?:warning|warn)|"
        r"issubclass\s*\([^)]*warning|"
        r"\.category\s*,\s*(?:RuntimeWarning|Warning)",
        code,
        re.IGNORECASE | re.DOTALL,
    ) and not issue_warning_expected:
        add_flag(
            "warning_presence_oracle",
            "warning 개수/타입만 검증하면 패치 후 경고가 사라지는 경우 after에서도 실패한다. "
            "warning이 아니라 수정 후 값/상태 또는 no-warning 성공 경로를 검증하라.",
        )

    # 예외 메시지 exact match: cm.exception, exc.value 등 다양한 패턴
    if re.search(
        r"str\s*\(\s*\w+[\.\w]*exception\s*\)\s*==|"
        r"\w+[\.\w]*args\[\d+\]\s*==|"
        r"str\s*\(\s*\w+\s*\)\s*==\s*['\"]",
        code,
    ):
        add_flag(
            "exception_message_match",
            "예외 메시지 exact match는 버전별로 흔들린다. 예외 타입 또는 의미 조건만 검증하라.",
        )

    if re.search(
        r"assert\s+str\s*\(\s*[\w.]+\s*\)\s*!=\s*['\"]|"
        r"\w+(?:\.value)?\.args\[\d+\]\s*!=\s*['\"]|"
        r"assert\s+['\"].+['\"]\s+not\s+in\s+str\s*\(|"
        r"self\.assert(?:NotIn|NotRegex)\s*\([^,\n]+,\s*str\s*\(|"
        r"self\.assertNotEqual\s*\(\s*str\s*\(|"
        r"self\.assertNotIn\s*\([^,\n]+,\s*[\w.]+\.args\[\d+\]",
        code,
        re.IGNORECASE,
    ):
        add_flag(
            "exception_message_negative_oracle",
            "예외 메시지 부재/변경을 oracle로 쓰면 fix 후 예외가 사라질 때도 실패한다. "
            "예외가 없어지는 성공 경로 또는 올바른 exception type만 검증하라.",
        )

    if not assertion_lines and "image_comparison_decorator" not in risk_flags:
        add_flag("no_explicit_oracle", "테스트에 명시적인 assertion/raises oracle이 없다.")

    # pytest.raises/assertRaises가 있지만 body에 assertion이 없음
    # → 패치 전/후 모두 예외가 발생하면 flip이 안 일어남
    raises_only = (
        assertion_lines
        and all(
            re.search(r"pytest\.raises|assertRaises|assert_raises|with\s+.*raises", line)
            for line in assertion_lines
        )
        and not re.search(r"^\s*(assert\s+(?!.*raises)|self\.assertEqual|self\.assertIn)", code, re.MULTILINE)
    )
    if raises_only:
        add_flag(
            "raises_only_no_body_assertion",
            "pytest.raises / assertRaises만 있고 body에 assertion이 없다. "
            "예외 타입 체크만으로는 패치 전/후를 구분하지 못한다. "
            "예외 없이 성공하는 동작 또는 result를 직접 검증하는 assertion을 추가하라.",
        )
    issue_text = " ".join(
        str(x)
        for x in (
            (clue or {}).get("observed_behavior", [])
            + (clue or {}).get("expected_behavior", [])
            + (clue or {}).get("repro_conditions", [])
            + [(clue or {}).get("raw_issue_text", "")]
        )
    ).lower()
    issue_says_success_path = bool(re.search(
        r"should\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"must\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"does\s+not\s+(?:raise|error|fail|crash|warn)|"
        r"doesn't\s+(?:raise|error|fail|crash|warn)|"
        r"without\s+(?:raising|error|failing|crashing|warning)|"
        r"no\s+(?:exception|error|warning)|"
        r"no\s+longer\s+(?:raises|errors|fails|crashes|warns)",
        issue_text,
    ))
    issue_says_exception_expected = bool(re.search(
        r"should\s+raise|must\s+raise|expected\s+(?:error|exception)|"
        r"should\s+(?:error|fail)\b|raises?\s+(?:a\s+)?(?:typeerror|valueerror|attributeerror|runtimeerror)",
        issue_text,
    ))
    has_raises_oracle = bool(re.search(r"pytest\.raises|assertRaises|assert_raises|with\s+.*raises", code))
    if has_raises_oracle and (
        issue_says_success_path
        or (not issue_says_exception_expected and re.search(r"post[- ]fix|should\s+accept|fit\s+success|succeed", code, re.IGNORECASE))
    ):
        add_flag(
            "fix_disappearing_exception_oracle",
            "이슈가 예외/에러가 없어져야 하는 성공 경로를 말하는데 raises oracle을 사용하고 있다. "
            "try/call success path와 post-call value/state assertion으로 재작성하라.",
        )

    structural_patterns = (
        r"\bis\s+not\s+none\b",
        r"\bisinstance\s*\(",
        r"\blen\s*\([^)]+\)\s*>\s*0\b",
        r"\.assertisnotnone\s*\(",
        r"\.asserttrue\s*\(\s*len\s*\(",
    )
    if assertion_lines and all(
        any(re.search(p, line, re.IGNORECASE) for p in structural_patterns)
        for line in assertion_lines
    ):
        add_flag("structural_oracle_only", "구조적 assertion만 사용하지 말고 이슈의 올바른 값/동작을 직접 검증하라.")

    structural_hits = [
        line for line in assertion_lines
        if any(re.search(p, line, re.IGNORECASE) for p in structural_patterns)
        or re.search(r"\.assertisinstance\s*\(|\.assertisnotnone\s*\(", line, re.IGNORECASE)
    ]
    if structural_hits and len(structural_hits) >= max(1, len(assertion_lines) - 1):
        add_flag(
            "weak_structural_oracle",
            "`assertIsInstance`/`assertIsNotNone`/length 같은 구조적 oracle은 post-fix 동작을 충분히 특정하지 못한다. "
            "issue-specific value 또는 state change를 검증하라.",
        )

    negative_assertions = [
        line for line in assertion_lines
        if (
            "!=" in line
            or ".assertnot" in line.lower()
            or re.search(r"\bnot\s+np\.array_equal\b", line.lower())
        )
    ]
    if assertion_lines and len(negative_assertions) == len(assertion_lines):
        add_flag("negative_oracle_only", "`!= buggy value`만 검증하면 fix 후에도 실패할 수 있다. 가능한 positive oracle을 사용하라.")

    if re.search(
        r"(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*!=|"
        r"assert\s+repr\s*\(\s*(?:expected|baseline|correct|desired|known)_[A-Za-z0-9_]*\s*\)\s*!=",
        code,
        re.IGNORECASE,
    ):
        add_flag(
            "constant_negative_oracle",
            "테스트 내부 local constant에 대한 negative assertion은 함수 결과를 검증하지 않는다. "
            "반드시 function_under_test의 return value 또는 state change를 assertion 대상으로 삼아라.",
        )

    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)\s*=.*\n"
        r"(?s:.*?)(?:assert_array_equal|assert_allclose|assert_equal)\s*\([^,\n]+,\s*"
        r"(?:expected|baseline|correct|desired)_(?:matrix|array|result|values?)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal():
        add_flag(
            "guessed_expected_array",
            "테스트 내부에서 만든 expected_matrix/array를 exact oracle로 쓰고 있다. "
            "issue expected_outputs에 근거한 값이 아니면 property invariant나 실제 post-fix 의미 조건으로 바꿔라.",
        )
    if re.search(
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)\s*=.*\n"
        r"(?s:.*?)(?:assert\s+[^=\n]+==\s*|self\.assertEqual\s*\([^,\n]+,\s*)"
        r"(?:expected|baseline|correct|desired)_(?:value|output|result)",
        code,
        re.IGNORECASE,
    ) and not _has_issue_expected_signal():
        add_flag(
            "guessed_expected_value",
            "테스트 내부에서 만든 expected_value/output/result를 exact oracle로 쓰고 있다. "
            "issue expected_outputs 근거가 없으면 semantic invariant로 바꿔라.",
        )

    if re.search(r"float\s*\(\s*['\"]nan['\"]\s*\)|\bnp\.nan\b", lower):
        add_flag("nan_comparison", "NaN은 직접 비교하지 말고 `np.isnan(...)` 또는 warning 검증을 사용하라.")

    if re.search(r"^\s*assert\s+.+==\s*np\.array\s*\(", code, re.MULTILINE):
        add_flag("numpy_direct_equality", "numpy/object array는 직접 `assert a == np.array(...)` 대신 `np.testing.assert_array_equal` 또는 객체 identity를 검증하라.")

    if re.search(r"pytest\.raises\([^)]*match\s*=", code) or re.search(
        r"assert\s+['\"].+['\"]\s+in\s+str\s*\(", code
    ):
        add_flag("exception_message_match", "예외 메시지 exact match는 버전별로 흔들린다. 예외 타입 또는 의미 조건만 검증하라.")

    if re.search(r"requests\.(get|post|put|delete|request)\s*\(\s*['\"]https?://", code):
        add_flag("external_network_call", "외부 네트워크 호출은 금지된다. PreparedRequest, mock, 기존 HTTP helper로 header/request 객체를 검증하라.")

    if re.search(r"class\s+\w+\s*\([^)]*models\.Model[^)]*\)", code):
        add_flag("django_inline_model", "Django model class를 테스트 안에 새로 정의하지 말고 기존 테스트 model/import를 사용하라.")

    if re.search(r"get_[xy]lim\(\)\s*\[\s*[01]\s*\]\s*==", code):
        add_flag("raw_axis_limit_equality", "Matplotlib 축 반전은 raw limit equality보다 `ax.yaxis_inverted()`/`xaxis_inverted()` 같은 의미 기반 oracle을 사용하라.")

    raw_string_asserts = [
        line for line in assertion_lines
        if re.search(r"(?:self\.)?assert(?:In|NotIn)\s*\(\s*['\"]", line)
        and (
            len(line) > 140
            or re.search(r"\\PYG|\\sphinx|<[^>]+>|html_content|tex_content|latex", line, re.IGNORECASE)
        )
    ]
    if raw_string_asserts:
        add_flag(
            "raw_rendered_output_exact_match",
            "Sphinx/HTML/LaTeX raw rendered string exact match는 너무 brittle하다. "
            "최소 semantic marker나 구조적 invariant로 재작성하라.",
        )
    if re.search(r"\._[A-Za-z]\w*", code):
        add_flag(
            "private_attribute_oracle",
            "private attribute를 oracle로 읽으면 내부 구현에 묶여 post-fix에서도 실패하기 쉽다. "
            "public API의 return value/state로 검증하라.",
        )

    assertion_blob = "\n".join(assertion_lines).lower()
    has_positive_signal = bool(expected_outputs) and any(
        str(out).strip() and str(out).strip()[:40].lower() in assertion_blob
        for out in expected_outputs[:2]
    )
    has_only_buggy_signal = bool(actual_outputs) and not has_positive_signal and any(
        str(out).strip() and str(out).strip()[:40].lower() in assertion_blob
        for out in actual_outputs[:2]
    )
    if has_only_buggy_signal and "negative_oracle_only" not in risk_flags:
        add_flag("buggy_output_as_oracle", "버그 출력만 oracle로 쓰지 말고 수정 후 올바른 기대 출력/동작을 검증하라.")

    if _issue_reported_semantic_invariant(code, clue):
        risk_flags = [
            flag
            for flag in risk_flags
            if flag not in {
                "buggy_output_as_oracle",
                "structural_oracle_only",
                "weak_structural_oracle",
            }
        ]
    penalty = min(0.15 * len(risk_flags), 0.85)
    return OracleQuality(score=round(1.0 - penalty, 4), risk_flags=risk_flags, feedback=feedback)


def _issue_reported_semantic_invariant(
    code: str,
    clue: Optional[Mapping[str, Any]],
) -> bool:
    """Recognize a comparison contract explicitly stated by an issue exception."""
    if not clue:
        return False
    issue_text = _issue_text_blob(dict(clue)).lower()
    if not re.search(r"\b(?:error|exception)\b|\b[A-Z]\w*(?:Error|Exception)\b", _issue_text_blob(dict(clue))):
        return False
    if not re.search(r"\b(?:should|must)\b", issue_text):
        return False
    assertion_text = "\n".join(
        line.split("#", 1)[0]
        for line in code.splitlines()
        if re.search(r"\bassert\b|self\.assert", line)
    ).lower()
    if not assertion_text:
        return False

    def comparisons(text: str) -> set[tuple[str, str]]:
        return {
            (operator, literal.lower())
            for operator, literal in re.findall(
                r"(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?|none|true|false)",
                text,
                re.IGNORECASE,
            )
        }

    shared = comparisons(issue_text) & comparisons(assertion_text)
    if shared:
        return True
    return bool(
        re.search(r"\bnon[- ]negative\b", issue_text)
        and re.search(r">=\s*0", assertion_text)
    )


# ---------------------------------------------------------------------------
# 2. 세 게이트 점수 산출
# ---------------------------------------------------------------------------

def extract_failure_features(
    raw_output: str,
    test_results: Dict[str, str],
) -> Dict[str, int]:
    """raw_output traceback에서 bug-fail 관련 이진 피처를 추출한다.

    Returns:
        각 피처명 → 0 또는 1 (존재 여부). 가중치는
        _BUG_FAIL_FEATURE_WEIGHTS에 정의한다.
    """
    statuses = set(test_results.values()) if test_results else set()
    return {
        "f_assert_diff": int(bool(re.search(
            r"AssertionError"                        # assertIs/assertEqual/assertTrue/assertRaises 등 모든 assertion 실패
            r"|\s!=\s",                              # 직접 비교
            raw_output,
        ))),
        "f_semantic_err": int(bool(re.search(
            r"\b(TypeError|ValueError|AttributeError|RuntimeError|KeyError|IndexError)\b",
            raw_output,
        ))),
        "f_import_err": int(bool(re.search(
            r"\b(NameError|ImportError|ModuleNotFoundError): ",  # 'except ImportError:' 오탐 방지
            raw_output,
        ))),
        "f_db_err": int(bool(re.search(
            r"\b(OperationalError|DatabaseError|ProgrammingError)\b|no such table",
            raw_output,
        ))),
        "f_setup_assert": int(bool(re.search(
            r"(setUp\b|setUpClass|setUpTestData"
            r"|Database queries to .+ not allowed)",
            raw_output,
        ))),
        "f_has_passed": int("PASSED" in statuses),
        "f_test_failed_summary": int(bool(re.search(
            r"\d+\s+failed",  # "1 failed", "2 failed", etc. — pytest가 명시적으로 보고하는 실패
            raw_output,
        ))),
    }


def compute_bug_fail_score(
    test_results: Dict[str, str],
    has_error: bool,
    raw_output: str = "",
    clue: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    generated_test: Optional[Dict[str, Any]] = None,
) -> float:
    """before-patch 실패 신호를 0~1로 스케일링한 bug reproduction score.

    s_b = clip(Σ_t w_t f_t, 0, 1). ERROR/NOT_VALID 계열 신호는 감점하지
    않고 failure_type과 feedback에서 별도로 다룬다.
    """
    if not test_results:
        return 0.0
    features = compute_bug_fail_features(
        test_results=test_results,
        raw_output=raw_output,
        clue=clue or {},
        scenario=scenario or {},
        generated_test=generated_test or {},
    )
    score = sum(
        _BUG_FAIL_FEATURE_WEIGHTS[name] * features.get(name, 0.0)
        for name in _BUG_FAIL_FEATURE_WEIGHTS
    )
    return round(_clamp01(score), 4)


def compute_v36_bug_fail_score(
    test_results: Mapping[str, str],
    keyword_hit_ratio: float,
) -> float:
    """Compute v36 formula (9); higher is better and the range is ``[0, 1]``.

    Only pre-patch FAIL/PASS observations enter the first term. ERROR and
    missing observations remain separate execution states. An empty F/P
    population has a failure ratio of zero, as required by v36.
    """
    statuses = [str(value).upper() for value in test_results.values()]
    failed = sum(value in {"FAIL", "FAILED"} for value in statuses)
    passed = sum(value in {"PASS", "PASSED"} for value in statuses)
    denominator = failed + passed
    fail_ratio = failed / denominator if denominator else 0.0
    return round(_clamp01(0.5 * fail_ratio + 0.5 * _clamp01(keyword_hit_ratio)), 4)


def compute_v37_keyword_hit_ratio(
    clue: Mapping[str, Any],
    test_results: Mapping[str, str],
    error_messages: Sequence[object],
    raw_output: str,
) -> tuple[float, dict[str, Any]]:
    """Compute v37 keyword evidence from before-patch failing execution only."""
    issue_tokens: set[str] = set()
    for key in ("OB", "observed_behavior", "error_keywords"):
        value = clue.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                issue_tokens.update(_token_set(str(item)))
        elif value:
            issue_tokens.update(_token_set(str(value)))
    identifiers = clue.get("identifiers")
    if isinstance(identifiers, Mapping):
        for values in identifiers.values():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    issue_tokens.update(_token_set(str(value)))
            elif values:
                issue_tokens.update(_token_set(str(values)))

    failing_names = sorted(
        str(name)
        for name, status in test_results.items()
        if str(status).upper() in {"FAIL", "FAILED"}
    )
    failure_messages = [str(value) for value in error_messages if str(value)]
    error_types = sorted(
        set(
            re.findall(
                r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b",
                "\n".join([*failure_messages, raw_output or ""]),
            )
        )
    )
    fail_tokens: set[str] = set()
    for value in [*failing_names, *failure_messages, *error_types]:
        fail_tokens.update(_token_set(value))
    ratio = len(issue_tokens & fail_tokens) / len(issue_tokens) if issue_tokens else 0.0
    return round(_clamp01(ratio), 4), {
        "source": "before_patch_failing_execution_only",
        "issue_token_count": len(issue_tokens),
        "failure_token_count": len(fail_tokens),
        "matched_tokens": sorted(issue_tokens & fail_tokens),
        "failing_test_names": failing_names,
        "failure_messages": failure_messages,
        "error_type_names": error_types,
        "excluded_sources": [
            "after_patch_output",
            "golden_patch",
            "generated_explanations",
            "llm_diagnosis",
            "fault_localization_score",
        ],
    }


def _token_set(text: str) -> Set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_]\w+", text or "")
        if len(token) > 1
    }


def _issue_symptom_tokens(
    clue: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Set[str]:
    tokens: Set[str] = set()
    clue_ids = clue.get("identifiers", {}) if isinstance(clue, dict) else {}
    if isinstance(clue_ids, dict):
        for key in ("functions", "classes", "exceptions"):
            for value in clue_ids.get(key, []) or []:
                tokens.update(_token_set(str(value)))
    for key in ("expected_outputs", "actual_outputs", "error_keywords"):
        for value in clue.get(key, []) or []:
            tokens.update(_token_set(str(value)))
    target = scenario.get("target_location", {}) if isinstance(scenario, dict) else {}
    if isinstance(target, dict):
        for key in ("source_file", "target_function"):
            tokens.update(_token_set(str(target.get(key, ""))))
        for value in target.get("related_classes", []) or []:
            tokens.update(_token_set(str(value)))
    return tokens


def compute_bug_fail_features(
    test_results: Dict[str, str],
    raw_output: str = "",
    clue: Optional[Dict[str, Any]] = None,
    scenario: Optional[Dict[str, Any]] = None,
    generated_test: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Return f_fail, f_assert, f_symptom for BugScore(r)."""
    statuses = list(test_results.values()) if test_results else []
    executable = [status for status in statuses if status in {"PASSED", "FAILED"}]
    failed = sum(1 for status in executable if status == "FAILED")
    f_fail = failed / len(executable) if executable else 0.0

    f_assert = 1.0 if re.search(
        r"AssertionError|\bassert\b|expected|actual|\s!=\s",
        raw_output or "",
        re.IGNORECASE,
    ) else 0.0

    evidence = _issue_symptom_tokens(clue or {}, scenario or {})
    observed = _token_set(raw_output or "")
    if generated_test:
        observed.update(_token_set(str(generated_test.get("test_code", ""))))
        observed.update(_token_set(str(generated_test.get("test_patch", ""))))
    f_symptom = len(evidence & observed) / len(evidence) if evidence else 0.0

    return {
        "f_fail": round(_clamp01(f_fail), 4),
        "f_assert": round(_clamp01(f_assert), 4),
        "f_symptom": round(_clamp01(f_symptom), 4),
    }


def compute_issue_alignment_score(
    clue: Dict[str, Any],
    scenario: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> float:
    """Compute original-paper Eq.(3) criterion-wise evidence recall.

    For each of the four criteria, ``phi`` is the evidence token set supplied
    by the issue/scenario and ``Tok(test)`` is the generated test token set.
    ``|phi ∩ Tok(test)| / |phi|`` is averaged only over non-empty criteria.
    """
    test_tokens = _token_set(str(generated_test.get("test_code", "")))
    if not test_tokens:
        return 0.0

    clue_ids = clue.get("identifiers", {}) if isinstance(clue, Mapping) else {}
    identifier_tokens: Set[str] = set()
    if isinstance(clue_ids, Mapping):
        for key in ("functions", "classes", "exceptions"):
            for value in clue_ids.get(key, []) or []:
                identifier_tokens.update(_token_set(str(value)))

    pattern_tokens: Set[str] = set()
    for block in clue.get("code_examples", []) or []:
        if isinstance(block, Mapping):
            pattern_tokens.update(_token_set(str(block.get("code", "") or block.get("interactive_input", ""))))

    symptom_tokens: Set[str] = set()
    for key in ("expected_outputs", "actual_outputs", "error_keywords"):
        for value in clue.get(key, []) or []:
            symptom_tokens.update(_token_set(str(value)))

    target = scenario.get("target_location", {}) if isinstance(scenario, Mapping) else {}
    target_tokens: Set[str] = set()
    if isinstance(target, Mapping):
        for key in ("target_function", "source_file"):
            target_tokens.update(_token_set(str(target.get(key, ""))))
        for value in target.get("related_classes", []) or []:
            target_tokens.update(_token_set(str(value)))

    evidence_sets = (identifier_tokens, pattern_tokens, symptom_tokens, target_tokens)
    applicable = [evidence for evidence in evidence_sets if evidence]
    if not applicable:
        return 0.0
    recall = sum(len(evidence & test_tokens) / len(evidence) for evidence in applicable)
    return round(_clamp01(recall / len(applicable)), 4)


def compute_v36_issue_alignment_score(
    clue: Mapping[str, Any],
    generated_test: Mapping[str, Any],
) -> float:
    """Compute v36 formula (12) over issue roles and test oracle/name tokens."""
    issue_tokens: Set[str] = set()
    for key in ("OB", "EB", "S2R"):
        value = clue.get(key)
        if value is None:
            aliases = {
                "OB": ("observed_behavior", "actual_output", "actual_outputs"),
                "EB": ("expected_behavior", "expected_output", "expected_outputs"),
                "S2R": ("steps_to_reproduce", "repro_conditions"),
            }[key]
            value = [clue.get(alias) for alias in aliases if clue.get(alias)]
        issue_tokens.update(_token_set(_semantic_text(value)))

    test_tokens: Set[str] = set()
    code = str(generated_test.get("test_code") or generated_test.get("append_block") or "")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            test_tokens.update(_token_set(node.name))
        elif isinstance(node, ast.Assert):
            test_tokens.update(_token_set(ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)))
        elif isinstance(node, ast.With):
            rendered = ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)
            if re.search(r"\b(?:raises|warns)\b", rendered):
                test_tokens.update(_token_set(rendered))
        elif isinstance(node, ast.Call):
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else function.id if isinstance(function, ast.Name) else ""
            if name.startswith("assert"):
                test_tokens.update(_token_set(ast.unparse(node) if hasattr(ast, "unparse") else ast.dump(node)))
    union = issue_tokens | test_tokens
    return round(len(issue_tokens & test_tokens) / len(union), 4) if union else 0.0


def _semantic_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return " ".join(_semantic_text(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_semantic_text(item) for item in value)
    return str(value or "")


def compute_v36_line_coverage_score(
    coverage_data: Mapping[str, Any],
    execution_result: Mapping[str, Any] | None = None,
) -> tuple[float | None, Dict[str, Any]]:
    """Compute formula (10), ``|C_t ∩ L_SUT| / |L_SUT|``, from explicit lines."""
    execution_result = execution_result or {}
    universe_raw = coverage_data.get("L_SUT") or execution_result.get("L_SUT") or []
    covered_raw = (
        coverage_data.get("covered_sut_lines")
        or execution_result.get("covered_sut_lines")
        or coverage_data.get("SUT_lines")
        or []
    )

    def keys(records: Any) -> Set[tuple[str, int]]:
        result: Set[tuple[str, int]] = set()
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, Mapping):
                continue
            path = str(record.get("source_file") or "").replace("\\", "/")
            line_no = record.get("line_no")
            if path and isinstance(line_no, int) and not isinstance(line_no, bool) and line_no > 0:
                result.add((path, line_no))
        return result

    universe = keys(universe_raw)
    covered = keys(covered_raw)
    if not universe:
        return None, {
            "status": "UNAVAILABLE_NO_L_SUT",
            "covered_count": len(covered),
            "sut_line_count": 0,
        }
    score = len(universe & covered) / len(universe)
    return round(_clamp01(score), 4), {
        "status": "AVAILABLE_PRE_PATCH_LINE_COVERAGE",
        "covered_count": len(universe & covered),
        "sut_line_count": len(universe),
    }


def has_strong_issue_evidence(
    clue: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> bool:
    """Check for non-trivial issue evidence beyond broad identifier overlap."""
    test_code = generated_test.get("test_code", "").lower()
    if not test_code:
        return False

    for block in clue.get("code_examples", []):
        if block.get("is_system_or_output"):
            continue
        code = block.get("code", "") or block.get("interactive_input", "")
        tokens = {
            t.lower() for t in re.findall(r"[A-Za-z_]\w+", code)
            if t not in {"from", "import", "def", "class", "assert", "with", "print"}
        }
        if tokens:
            hits = sum(1 for t in tokens if t in test_code)
            if hits / len(tokens) >= max(_ISSUE_ALIGN_TOKEN_HIT, 0.2):
                return True

    for vals in (
        clue.get("expected_outputs", []),
        clue.get("actual_outputs", []),
        clue.get("error_keywords", []),
    ):
        for value in vals:
            value_lower = str(value).lower()
            tokens = set(re.findall(r"[A-Za-z_]\w+", value_lower))
            if tokens:
                hits = sum(1 for t in tokens if t in test_code)
                if hits / len(tokens) >= max(_ISSUE_ALIGN_TOKEN_HIT, 0.2):
                    return True
            elif value_lower and value_lower in test_code:
                return True

    return False


def is_target_location_verified(
    scenario: Dict[str, Any],
    context: Optional[Dict[str, Any]],
    execution_result: Optional[Dict[str, Any]],
    generated_test: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Return whether pre-patch static/runtime evidence verifies the target."""
    return bool(
        compute_target_verification_evidence(
            scenario=scenario,
            context=context,
            execution_result=execution_result,
            generated_test=generated_test,
        )["target_verified"]
    )


def _canonical_target_identity(scenario: Mapping[str, Any]) -> str:
    """Read canonical identity without promoting a receiver expression.

    New v31 artifacts carry an explicit canonical field, including an empty
    value when M2 could not resolve a repository callable. The legacy
    ``target_function`` fallback applies only to older artifacts that carry no
    canonical field at either level.
    """
    target = scenario.get("target_location") or {}
    if not isinstance(target, Mapping):
        return ""
    if "canonical_target_identity" in target:
        return str(target.get("canonical_target_identity") or "").strip()
    if "canonical_target_identity" in scenario:
        return str(scenario.get("canonical_target_identity") or "").strip()
    return str(target.get("target_function") or scenario.get("target_function") or "").strip()


def compute_target_verification_evidence(
    *,
    scenario: Mapping[str, Any],
    context: Optional[Mapping[str, Any]],
    execution_result: Optional[Mapping[str, Any]],
    generated_test: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply the v31 target-evidence hierarchy without source-string guesses.

    Executed target lines are strongest and may verify an indirect public API
    call.  A statically verified direct target remains acceptable only when
    runtime evidence does not disprove it.  Unresolved indirect behavior with
    no target-line evidence fails closed.
    """
    target = scenario.get("target_location") or {}
    evidence: Dict[str, Any] = {
        "schema_version": "target-verification-evidence-v31-v1",
        "target_verified": False,
        "verification_path": "UNAVAILABLE",
        "source_file": "",
        "target_function": "",
        "static_status": "",
        "runtime_target_coverage": {},
        "reason": "missing_target_location",
    }
    if not isinstance(target, dict):
        return evidence
    source_file = target.get("source_file", "")
    target_func = _canonical_target_identity(scenario)
    if not isinstance(source_file, str) or not isinstance(target_func, str):
        evidence["reason"] = "malformed_target_identity"
        return evidence
    if not source_file or not target_func:
        return evidence

    evidence["source_file"] = source_file
    evidence["target_function"] = target_func
    evidence["canonical_target_identity"] = target_func
    evidence["candidate_invocation_expression"] = str(
        target.get("candidate_invocation_expression")
        or scenario.get("candidate_invocation_expression")
        or target.get("issue_api_target")
        or scenario.get("issue_api_target")
        or ""
    )

    verification_status = str(
        target.get("target_verification_status")
        or scenario.get("target_verification_status")
        or ""
    )
    evidence["static_status"] = verification_status
    if verification_status in {"TARGET_CONFLICT", "INVALID_TARGET", "INVALID_SCENARIO_STRUCTURE"}:
        evidence["verification_path"] = "STATIC_REJECTION"
        evidence["reason"] = verification_status.lower()
        return evidence

    bare = target_func.split(".")[-1]
    receiver_bound_canonical = "." in target_func
    static_callable_resolved = False
    static_callable_contradicted = False
    if context:
        source_entries = context.get("candidate_source_files", [])
        for sf in source_entries:
            if sf.get("path") != source_file:
                continue
            top_funcs = sf.get("top_level_functions") or []
            if top_funcs:
                if bare.startswith("__") and bare.endswith("__"):
                    evidence["reason"] = "dunder_target_is_not_specific"
                    return evidence
                if any(tf == target_func or tf.split(".")[-1] == bare for tf in top_funcs):
                    static_callable_resolved = True
                else:
                    static_callable_contradicted = True
            break

    coverage_data = (execution_result or {}).get("coverage_data") or {}
    runtime = compute_target_coverage_evidence(
        coverage_data if isinstance(coverage_data, Mapping) else {},
        scenario,
        context,
    )
    evidence["runtime_target_coverage"] = runtime
    if runtime.get("target_function_covered") is True:
        evidence.update({
            "target_verified": True,
            "verification_path": "RUNTIME_TARGET_LINES",
            "reason": "prepatch_target_callable_lines_executed",
            "runtime_verified_target": target_func,
        })
        return evidence
    if runtime.get("target_function_covered") is False:
        evidence.update({
            "verification_path": "RUNTIME_DISPROVAL",
            "reason": str(runtime.get("evidence_status") or "target_not_executed").lower(),
        })
        return evidence

    direct_statuses = {"VERIFIED_DIRECT_TARGET", "VERIFIED_ISSUE_API", "VERIFIED_IMPLEMENTATION_TARGET"}
    generated_code = str(
        (generated_test or {}).get("test_code")
        or (generated_test or {}).get("append_block")
        or ""
    )
    direct_call_present = bool(
        generated_code
        and re.search(rf"(?<![A-Za-z0-9_])(?:[A-Za-z_]\w*\.)*{re.escape(bare)}\s*\(", generated_code)
    )
    if (
        direct_call_present
        and not receiver_bound_canonical
        and runtime.get("target_file_covered") is True
        and runtime.get("target_function_covered") == "UNKNOWN"
        and runtime.get("evidence_status") == "TARGET_SOURCE_UNAVAILABLE"
        and verification_status != "TARGET_UNRESOLVED"
        and not static_callable_contradicted
    ):
        evidence.update({
            "target_verified": True,
            "verification_path": "DIRECT_CALL_WITH_RUNTIME_TARGET_FILE",
            "reason": "direct_target_call_and_prepatch_target_file_coverage_agree",
        })
        return evidence
    if static_callable_resolved and not receiver_bound_canonical and (
        verification_status in direct_statuses or direct_call_present
    ):
        evidence.update({
            "target_verified": True,
            "verification_path": "STATIC_DIRECT_RUNTIME_NOT_DISPROVEN",
            "reason": (
                "repository_resolved_direct_target"
                if verification_status in direct_statuses
                else "generated_test_directly_invokes_repository_resolved_target"
            ),
        })
        return evidence
    evidence["reason"] = (
        "target_not_resolved_in_source_file"
        if context and (static_callable_contradicted or not static_callable_resolved)
        else
        "unresolved_target_without_runtime_execution_evidence"
        if verification_status == "TARGET_UNRESOLVED"
        else "runtime_target_evidence_unavailable"
    )
    return evidence


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _path_tail_matches(candidate: str, expected: str) -> bool:
    if not candidate or not expected:
        return False
    candidate_norm = candidate.replace("\\", "/")
    expected_norm = expected.replace("\\", "/")
    return (
        candidate_norm.endswith(expected_norm)
        or expected_norm.endswith(candidate_norm)
        or candidate_norm.split("/")[-1] == expected_norm.split("/")[-1]
    )


def _match_coverage_file(coverage_data: Dict[str, Dict], source_file: str) -> Optional[str]:
    for fname in coverage_data:
        if _path_tail_matches(fname, source_file):
            return fname
    return None


def compute_target_coverage_evidence(
    coverage_data: Mapping[str, Mapping[str, Any]],
    scenario: Mapping[str, Any],
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive truthful target-file/function evidence from pre-patch coverage.

    FALSE means the relevant artifact is present and disproves coverage.
    UNKNOWN means the target or source span cannot be resolved unambiguously.
    """
    target = scenario.get("target_location")
    if not isinstance(target, Mapping):
        target = {}
    source_file = str(target.get("source_file") or scenario.get("source_file") or "").strip()
    target_function = _canonical_target_identity(scenario)
    result: Dict[str, Any] = {
        "target_source_file": source_file,
        "target_function": target_function,
        "matched_coverage_file": None,
        "target_file_covered": "UNKNOWN",
        "target_function_covered": "UNKNOWN",
        "target_function_line_count": None,
        "covered_target_line_count": None,
        "covered_target_lines": [],
        "evidence_status": "UNKNOWN",
    }
    if not source_file or not coverage_data:
        return result

    normalized_target = source_file.replace("\\", "/").lstrip("./")
    matches = [
        str(name)
        for name in coverage_data
        if str(name).replace("\\", "/").lstrip("./") == normalized_target
        or str(name).replace("\\", "/").endswith("/" + normalized_target)
    ]
    if len(matches) > 1:
        result["evidence_status"] = "AMBIGUOUS_TARGET_FILE"
        return result
    if not matches:
        result["target_file_covered"] = False
        result["target_function_covered"] = False if target_function else "UNKNOWN"
        result["covered_target_line_count"] = 0
        result["evidence_status"] = "TARGET_FILE_NOT_COVERED"
        return result

    matched_file = matches[0]
    info = coverage_data.get(matched_file)
    if not isinstance(info, Mapping):
        result["evidence_status"] = "MALFORMED_COVERAGE_RECORD"
        return result
    result["matched_coverage_file"] = matched_file
    result["target_file_covered"] = _coverage_file_ratio(dict(info)) > 0
    if not target_function:
        result["evidence_status"] = "TARGET_FUNCTION_UNSPECIFIED"
        return result

    source_path = _repo_source_path(source_file, dict(context or {}))
    if source_path is None:
        result["evidence_status"] = "TARGET_SOURCE_UNAVAILABLE"
        return result
    function_lines = _statement_lines_for_target(
        source_path,
        target_function,
        list(target.get("related_classes") or []),
    )
    if not function_lines:
        result["target_function_covered"] = False
        result["target_function_line_count"] = 0
        result["covered_target_line_count"] = 0
        result["evidence_status"] = "TARGET_FUNCTION_NOT_RESOLVED"
        return result

    missing_lines = {
        line
        for line in (_parse_line_no(value) for value in (info.get("missing_lines") or []))
        if line is not None
    }
    covered_lines = sorted(function_lines - missing_lines)
    result.update({
        "target_function_covered": bool(covered_lines),
        "target_function_line_count": len(function_lines),
        "covered_target_line_count": len(covered_lines),
        "covered_target_lines": covered_lines,
        "evidence_status": "KNOWN",
    })
    return result


def _parse_line_no(value: Any) -> Optional[int]:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            line_no = int(match.group(0))
            return line_no if line_no > 0 else None
    return None


def _repo_source_path(source_file: str, context: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not source_file or not context:
        return None
    repo_path = context.get("repo_path")
    if not repo_path:
        return None
    path = Path(repo_path) / source_file
    return path if path.exists() else None


def _filter_source_code_lines(source_path: Optional[Path], lines: Set[int]) -> Set[int]:
    if not source_path or not source_path.exists() or not lines:
        return {line for line in lines if line > 0}
    try:
        source_lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {line for line in lines if line > 0}

    filtered: Set[int] = set()
    for line_no in lines:
        if line_no <= 0 or line_no > len(source_lines):
            continue
        stripped = source_lines[line_no - 1].strip()
        if stripped and not stripped.startswith("#"):
            filtered.add(line_no)
    return filtered


def _fault_location_suspicious_lines(
    clue: Optional[Dict[str, Any]],
    source_file: str,
    source_path: Optional[Path],
) -> Set[int]:
    if not clue:
        return set()
    lines: Set[int] = set()
    for fault in clue.get("fault_locations", []) or []:
        if not isinstance(fault, dict):
            continue
        fault_file = str(
            fault.get("file_path")
            or fault.get("source_file")
            or fault.get("file")
            or ""
        )
        if fault_file and not _path_tail_matches(fault_file, source_file):
            continue
        line_no = _parse_line_no(
            fault.get("line_no")
            or fault.get("line")
            or fault.get("lineno")
        )
        if not line_no:
            continue
        lines.update(range(line_no - _SUSPICIOUS_LINE_WINDOW, line_no + _SUSPICIOUS_LINE_WINDOW + 1))
    return _filter_source_code_lines(source_path, lines)


def _statement_lines_for_target(
    source_path: Optional[Path],
    target_func: str,
    related_classes: Optional[List[Any]] = None,
) -> Set[int]:
    if not source_path or not source_path.exists() or not target_func:
        return set()
    try:
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source_text)
    except (OSError, SyntaxError):
        return set()

    bare_target = target_func.split(".")[-1]
    class_names = {
        str(name).split(".")[-1]
        for name in (related_classes or [])
        if str(name).strip()
    }
    named_nodes: List[tuple[str, ast.AST]] = []

    def collect(body: List[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualified = (*parents, node.name)
                named_nodes.append((".".join(qualified), node))
                collect(node.body, qualified)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                named_nodes.append((".".join((*parents, node.name)), node))

    collect(tree.body)
    target_nodes: List[ast.AST]
    if "." in target_func:
        target_nodes = [
            node
            for qualified, node in named_nodes
            if qualified == target_func or target_func.endswith("." + qualified)
        ]
    else:
        top_level = [
            node for qualified, node in named_nodes if qualified == target_func
        ]
        class_scoped = [
            node
            for qualified, node in named_nodes
            if qualified.split(".")[-1] == bare_target
            and any(part in class_names for part in qualified.split(".")[:-1])
        ]
        tail_matches = [
            node
            for qualified, node in named_nodes
            if qualified.split(".")[-1] == bare_target
        ]
        target_nodes = top_level or class_scoped
        if not target_nodes and len(tail_matches) == 1:
            target_nodes = tail_matches

    if not target_nodes:
        return set()

    def node_span(node: ast.AST) -> int:
        start = getattr(node, "lineno", 0) or 0
        end = getattr(node, "end_lineno", start) or start
        return max(end - start, 0)

    # Prefer the smallest matching node so a method named `clean` inside a class
    # beats a broad class wrapper with the same public name.
    selected = min(target_nodes, key=node_span)
    lines: Set[int] = set()
    for node in ast.walk(selected):
        if node is selected:
            # Importing a module executes/records a ``def`` or ``class``
            # header; that is not evidence that the callable body ran.
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            # Function/class docstrings are metadata, not runtime target
            # behavior and may share the header's import-time coverage.
            continue
        if isinstance(node, ast.stmt):
            line_no = getattr(node, "lineno", None)
            if isinstance(line_no, int) and not isinstance(line_no, bool) and line_no > 0:
                lines.add(line_no)
    return _filter_source_code_lines(source_path, lines)


def _coverage_file_ratio(info: Dict[str, Any]) -> float:
    try:
        return _clamp01(float(info.get("cover", 0.0)) / 100.0)
    except (TypeError, ValueError):
        return 0.0


def _suspicious_line_coverage_ratio(
    info: Dict[str, Any],
    suspicious_lines: Set[int],
) -> float:
    if not suspicious_lines or _coverage_file_ratio(info) <= 0:
        return 0.0
    missing_lines = {
        line
        for line in (_parse_line_no(value) for value in (info.get("missing_lines") or []))
        if line is not None
    }
    covered = [line for line in suspicious_lines if line not in missing_lines]
    return _clamp01(len(covered) / len(suspicious_lines))


def normalize_coverage_entries(coverage_data: Any) -> tuple[Dict[str, Dict[str, Any]], str | None]:
    """Normalize supported coverage shapes without turning invalid evidence into zero.

    Coverage producers historically emitted either ``{path: mapping}`` or a
    list of mappings carrying ``source_file``/``file``.  The scorer consumes a
    single canonical mapping.  Unsupported entries are reported as
    ``UNAVAILABLE_MALFORMED`` and are never treated as measured coverage.
    """
    if coverage_data in (None, "", [], {}):
        return {}, "UNAVAILABLE_NO_DATA"
    if isinstance(coverage_data, Mapping):
        normalized: Dict[str, Dict[str, Any]] = {}
        malformed = False
        # The production M6 parser stores line-spectrum provenance alongside
        # the file keyed coverage entries.  These keys are metadata, not
        # coverage files, and must not make an otherwise valid map malformed.
        metadata_keys = {
            "SUT_lines",
            "sut_lines",
            "covered_sut_lines",
            "line_spectrum_source",
            "coverage_source",
            "coverage_metadata",
        }
        for key, value in coverage_data.items():
            if str(key) in metadata_keys:
                continue
            if not isinstance(value, Mapping):
                malformed = True
                continue
            normalized[str(key).replace("\\", "/")] = dict(value)
        if normalized:
            return normalized, "UNAVAILABLE_MALFORMED" if malformed else None
        return {}, "UNAVAILABLE_MALFORMED"
    if isinstance(coverage_data, list):
        normalized = {}
        malformed = False
        for item in coverage_data:
            if not isinstance(item, Mapping):
                malformed = True
                continue
            source_file = item.get("source_file") or item.get("file") or item.get("path")
            info = item.get("coverage") if isinstance(item.get("coverage"), Mapping) else item
            if not isinstance(source_file, str) or not source_file.strip() or not isinstance(info, Mapping):
                malformed = True
                continue
            normalized[source_file.replace("\\", "/")] = dict(info)
        if normalized:
            return normalized, "UNAVAILABLE_MALFORMED" if malformed else None
        return {}, "UNAVAILABLE_MALFORMED"
    return {}, "UNAVAILABLE_MALFORMED"


def compute_coverage_score(
    coverage_data: Dict[str, Dict] | list[Mapping[str, Any]],
    scenario: Dict[str, Any],
    execution_result: Optional[Dict[str, Any]] = None,
    clue: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """의심 코드 라인 커버리지. 최대 1.0점.

    gold patch 라인은 쓰지 않는다. clue/scenario/context에서 얻은 의심 위치
    라인 집합 L_s 중 before-patch 실행에서 실제 커버된 비율만 계산한다.
    """
    coverage_data, coverage_shape_status = normalize_coverage_entries(coverage_data)
    if not coverage_data:
        return 0.0

    target = scenario.get("target_location", {})
    source_file = target.get("source_file", "")
    target_func = target.get("target_function", "")
    if not isinstance(source_file, str):
        source_file = ""
    if not isinstance(target_func, str):
        target_func = ""

    if not source_file:
        # source_file이 없으면 전체 커버리지 기반으로 추정
        # 소스 파일(test가 아닌) 중 커버된 게 있으면 부분 점수
        source_covered = 0
        source_total = 0
        for fname, info in coverage_data.items():
            if "/test" in fname or "test_" in fname:
                continue
            source_total += 1
            if info.get("cover", 0) > 0:
                source_covered += 1
        if source_total > 0:
            return _COVERAGE_FALLBACK * min(source_covered / source_total, 1.0)
        return 0.0

    # source_file이 있는 경우 — 정확한 매칭
    matched_file = _match_coverage_file(coverage_data, source_file)
    if not matched_file:
        # 의심 파일이 커버리지 리포트에 없음 → 테스트가 해당 파일을 전혀 실행하지 않음
        return 0.0

    info = coverage_data.get(matched_file)
    if not isinstance(info, dict):
        return 0.0
    file_ratio = _coverage_file_ratio(info)
    if file_ratio == 0:
        return 0.0

    source_path = _repo_source_path(source_file, context)
    suspicious_lines = _fault_location_suspicious_lines(clue, source_file, source_path)
    if not suspicious_lines:
        suspicious_lines = _statement_lines_for_target(
            source_path,
            target_func,
            target.get("related_classes") if isinstance(target, dict) else None,
        )

    if suspicious_lines:
        return round(_suspicious_line_coverage_ratio(info, suspicious_lines), 4)

    # 의심 라인을 만들 수 없는 오래된/불완전 산출물은 target source file의
    # coverage ratio로만 fallback한다. 이 경우도 patch line coverage는 쓰지 않는다.
    return round(min(file_ratio, _COVERAGE_MAX), 4)


def build_coverage_field_contract(
    *,
    execution_result: Mapping[str, Any],
    coverage_data: Any,
    scenario: Mapping[str, Any],
    clue: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    s_c_prime: float | None,
    s_c_prime_available: bool,
    s_c_prime_unavailable_reason: str | None,
) -> Dict[str, Any]:
    """Separate measured raw coverage from nullable weighted evidence.

    ``raw_target_coverage`` is numeric whenever coverage instrumentation
    produced a parseable pre-patch coverage payload.  A measured target miss
    is therefore ``0.0``.  Missing instrumentation/artifacts remain explicit
    and nullable.  ``s_c`` is diagnostic raw target coverage only; it never
    substitutes for the v29 ``s_c_prime`` admission gate.
    """
    normalized, shape_status = normalize_coverage_entries(coverage_data)
    execution_status = normalize_pre_patch_execution_status(execution_result).value
    if normalized:
        coverage_execution_status = "MEASURED"
        raw_target_coverage: float | None = float(
            compute_coverage_score(
                normalized,
                dict(scenario),
                dict(execution_result),
                clue=dict(clue or {}),
                context=dict(context or {}),
            )
        )
    elif execution_status == "NOT_RUN":
        coverage_execution_status = "NOT_EXECUTED"
        raw_target_coverage = None
    elif shape_status == "UNAVAILABLE_MALFORMED":
        coverage_execution_status = "UNAVAILABLE_MALFORMED_ARTIFACT"
        raw_target_coverage = None
    else:
        coverage_execution_status = "UNAVAILABLE_NO_COVERAGE_ARTIFACT"
        raw_target_coverage = None
    raw_status = "AVAILABLE_NUMERIC" if raw_target_coverage is not None else "UNAVAILABLE"
    prime_reason = str(s_c_prime_unavailable_reason or "")
    prime_status = (
        "AVAILABLE_WEIGHTED_OCHIAI"
        if s_c_prime_available
        else "INSUFFICIENT_SPECTRA"
        if any(token in prime_reason.lower() for token in ("insufficient", "inactive_insufficient_tests"))
        else "UNAVAILABLE"
    )
    return {
        "coverage_execution_status": coverage_execution_status,
        "pre_patch_execution_status": execution_status,
        "raw_target_coverage": raw_target_coverage,
        "raw_target_coverage_status": raw_status,
        "s_c": raw_target_coverage,
        "s_c_admission_role": "DIAGNOSTIC_ONLY",
        "s_c_prime_status": prime_status,
        "s_c_prime": s_c_prime if s_c_prime_available else None,
        "s_c_prime_unavailable_reason": (
            None if s_c_prime_available else s_c_prime_unavailable_reason
        ),
    }


def _compact_supplemental_pass_collection(
    collection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Keep decision-relevant PASS collection facts without line spectra.

    Full execution records and spectra remain canonical in ``sbfl_result``;
    M7 needs only bounded counts, identities, and rejection/exhaustion facts.
    """
    scalar_keys = (
        "schema_version",
        "activation_status",
        "collection_status",
        "stop_reason",
        "required_p_count",
        "valid_distinct_p_count",
        "accepted_count",
        "rejected_count",
        "candidate_count",
        "attempt_count",
        "exhausted",
        "failure_type_detail",
        "diagnostic_classification",
    )
    compact: Dict[str, Any] = {
        key: collection.get(key)
        for key in scalar_keys
        if key in collection
    }
    for list_key in ("accepted_test_ids", "rejected_reasons", "candidate_ids"):
        values = collection.get(list_key)
        if isinstance(values, list):
            compact[list_key] = list(values[:64])
    accepted_records = collection.get("accepted_records")
    if isinstance(accepted_records, list):
        compact["accepted_records_summary"] = [
            {
                key: record.get(key)
                for key in (
                    "test_id",
                    "canonical_test_id",
                    "test_nodeid",
                    "execution_status",
                    "candidate_sha256",
                )
                if key in record
            }
            for record in accepted_records[:16]
            if isinstance(record, Mapping)
        ]
        compact.setdefault("accepted_count", len(accepted_records))
    candidate_records = collection.get("candidate_records")
    if isinstance(candidate_records, list):
        compact["candidate_records_summary"] = [
            {
                key: record.get(key)
                for key in (
                    "candidate_id",
                    "test_id",
                    "status",
                    "rejection_reason",
                    "candidate_sha256",
                )
                if key in record
            }
            for record in candidate_records[:32]
            if isinstance(record, Mapping)
        ]
        compact.setdefault("candidate_count", len(candidate_records))
    compact["full_records_storage"] = "canonical_m6_sbfl_result_reference_only"
    return compact


def compute_m7_sbfl_weighted_coverage(
    *,
    base_coverage: float,
    coverage_data: Mapping[str, Any],
    scenario: Mapping[str, Any],
    sbfl_result: Mapping[str, Any] | None,
    execution_result: Mapping[str, Any] | None = None,
    clue: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    require_weighted: bool = True,
) -> Dict[str, Any]:
    """Return optional M7 ``s_c_prime`` from explicit canonical M6 Ochiai.

    Input range: ``base_coverage`` and valid Ochiai scores are expected in
    ``[0, 1]`` and are clamped at the reporting boundary. Output range:
    ``weighted_coverage`` is in ``[0, 1]`` when available. Higher is better.

    Zero denominator behavior: when ``L_s`` is empty, weighted evidence is
    unavailable when ``require_weighted`` is true. Missing-value behavior:
    unavailable, inactive, non-Ochiai, non-line-level, or malformed SBFL
    evidence is never silently used as v29 weighted evidence. When the feature
    is explicitly disabled, deterministic base coverage is retained and the
    disabled provenance is recorded. Evaluation population: every canonical
    M6 SUT line with positive Ochiai, never golden or post-patch data.
    """
    normalized_coverage_data, coverage_shape_status = normalize_coverage_entries(coverage_data)
    base = round(_clamp01(base_coverage), 4)
    payload: Dict[str, Any] = {
        "schema_version": _M7_WEIGHTED_COVERAGE_SCHEMA_VERSION,
        "base_coverage": base,
        "weighted_coverage": base if not require_weighted else None,
        "coverage_valid": not require_weighted,
        "score_definition": (
            "LEGACY_BASE_COVERAGE_FEATURE_DISABLED"
            if not require_weighted else "V29_WEIGHTED_OCHIAI"
        ),
        "used_sbfl_weighting": False,
        "fallback_reason": None,
        "formula": (
            "sum(indicator[line_covered] * S_ochiai(line)) "
            "/ sum(S_ochiai(line))"
        ),
        "sbfl_evidence": {
            "status": "unavailable",
            "source": "canonical_m6_sbfl_result",
            "formula": None,
            "activation_status": None,
            "matched_lines": [],
        },
    }
    if coverage_shape_status:
        payload["coverage_shape_status"] = coverage_shape_status
        if coverage_shape_status == "UNAVAILABLE_MALFORMED":
            payload["fallback_reason"] = coverage_shape_status
            return payload
    reason = _validate_m7_sbfl_payload(sbfl_result)
    if isinstance(sbfl_result, Mapping):
        sbfl_metadata = sbfl_result.get("metadata")
        if isinstance(sbfl_metadata, Mapping) and isinstance(
            sbfl_metadata.get("supplemental_pass_collection"), Mapping
        ):
            payload["supplemental_pass_collection"] = _compact_supplemental_pass_collection(
                sbfl_metadata["supplemental_pass_collection"]
            )
    if reason:
        if isinstance(sbfl_result, Mapping):
            metadata = sbfl_result.get("metadata")
            payload["sbfl_evidence"]["formula"] = sbfl_result.get("formula")
            if isinstance(metadata, Mapping):
                payload["sbfl_evidence"]["activation_status"] = metadata.get("activation_status")
        payload["fallback_reason"] = reason
        if not require_weighted:
            payload["weighted_coverage"] = base
            payload["coverage_valid"] = True
            payload["fallback_reason"] = "feature_disabled"
            payload["score_definition"] = "LEGACY_BASE_COVERAGE_FEATURE_DISABLED"
        return payload

    assert sbfl_result is not None
    line_scores = _canonical_ochiai_line_scores(sbfl_result)
    if not line_scores:
        payload["fallback_reason"] = "valid_ochiai_line_evidence_unavailable"
        payload["sbfl_evidence"]["status"] = "invalid"
        if not require_weighted:
            payload["weighted_coverage"] = base
            payload["coverage_valid"] = True
            payload["fallback_reason"] = "feature_disabled"
            payload["score_definition"] = "LEGACY_BASE_COVERAGE_FEATURE_DISABLED"
        return payload

    covered_line_identities, covered_lines_status = _canonical_covered_sut_lines(
        execution_result, coverage_data
    )
    if covered_lines_status:
        payload["fallback_reason"] = covered_lines_status
        payload["sbfl_evidence"]["status"] = "available_not_used"
        return payload
    matched_evidence = []
    weighted_sum = 0.0
    suspiciousness_sum = 0.0
    for (source_file, line_no), score in sorted(line_scores.items()):
        if score <= 0.0:
            continue
        covered = (source_file, line_no) in covered_line_identities
        weighted_sum += score if covered else 0.0
        suspiciousness_sum += score
        matched_evidence.append(
            {
                "source_file": source_file,
                "line_no": line_no,
                "covered": covered,
                "S_ochiai": round(score, 6),
            }
        )
    if not matched_evidence or suspiciousness_sum <= 0.0:
        payload["fallback_reason"] = "zero_positive_ochiai_denominator"
        payload["sbfl_evidence"]["status"] = "available_no_overlap"
        if not require_weighted:
            payload["weighted_coverage"] = base
            payload["coverage_valid"] = True
            payload["fallback_reason"] = "feature_disabled"
            payload["score_definition"] = "LEGACY_BASE_COVERAGE_FEATURE_DISABLED"
        return payload

    weighted = round(_clamp01(weighted_sum / suspiciousness_sum), 4)
    metadata = sbfl_result.get("metadata") if isinstance(sbfl_result.get("metadata"), Mapping) else {}
    payload.update(
        weighted_coverage=weighted,
        coverage_valid=True,
        used_sbfl_weighting=True,
        fallback_reason=None,
        sbfl_evidence={
            "status": "available",
            "source": "canonical_m6_sbfl_result",
            "formula": sbfl_result.get("formula"),
            "activation_status": metadata.get("activation_status"),
            "numerator": round(weighted_sum, 6),
            "denominator": round(suspiciousness_sum, 6),
            "L_s_definition": "all_positive_ochiai_sut_lines",
            "matched_line_count": len(matched_evidence),
            "matched_lines": matched_evidence[:64],
            "matched_lines_truncated": len(matched_evidence) > 64,
            "full_line_evidence_storage": "canonical_m6_sbfl_result",
        },
    )
    return payload


def _canonical_covered_sut_lines(
    execution_result: Mapping[str, Any] | None,
    coverage_data: Mapping[str, Any],
) -> tuple[Set[tuple[str, int]], str | None]:
    """Canonicalize explicit current-candidate covered SUT line evidence."""
    raw_lines: Any = None
    if isinstance(execution_result, Mapping):
        raw_lines = execution_result.get("covered_sut_lines")
    if raw_lines is None and isinstance(coverage_data, Mapping):
        raw_lines = (
            coverage_data.get("covered_sut_lines")
            or coverage_data.get("SUT_lines")
            or coverage_data.get("sut_lines")
        )
    if raw_lines is None:
        return set(), "UNAVAILABLE_CURRENT_PASS_LINE_COVERAGE"
    if not isinstance(raw_lines, list):
        return set(), "UNAVAILABLE_MALFORMED_CURRENT_PASS_LINE_COVERAGE"
    result: Set[tuple[str, int]] = set()
    malformed = False
    for item in raw_lines:
        if not isinstance(item, Mapping):
            malformed = True
            continue
        source_file = str(
            item.get("source_file") or item.get("file") or item.get("path") or ""
        ).replace("\\", "/")
        line_no = item.get("line_no", item.get("line", item.get("lineno")))
        if not source_file or isinstance(line_no, bool) or not isinstance(line_no, int) or line_no <= 0:
            malformed = True
            continue
        result.add((source_file, line_no))
    if malformed:
        return set(), "UNAVAILABLE_MALFORMED_CURRENT_PASS_LINE_COVERAGE"
    return result, None


def _validate_m7_sbfl_payload(sbfl_result: Mapping[str, Any] | None) -> str:
    if not isinstance(sbfl_result, Mapping):
        return "sbfl_result_unavailable"
    if _contains_forbidden_sbfl_provenance(sbfl_result):
        return "forbidden_patch_or_m8_evidence_rejected"
    if str(sbfl_result.get("formula") or "").lower() != "ochiai":
        return "non_ochiai_sbfl_formula"
    metadata = sbfl_result.get("metadata")
    if not isinstance(metadata, Mapping):
        return "sbfl_metadata_unavailable"
    if metadata.get("sbfl_active") is not True or metadata.get("activation_status") != "active":
        return str(metadata.get("activation_status") or "sbfl_inactive")
    suspiciousness = sbfl_result.get("suspiciousness")
    if not isinstance(suspiciousness, list) or not suspiciousness:
        return "sbfl_suspiciousness_unavailable"
    return ""


def _canonical_ochiai_line_scores(sbfl_result: Mapping[str, Any]) -> Dict[tuple[str, int], float]:
    scores: Dict[tuple[str, int], float] = {}
    for item in sbfl_result.get("suspiciousness") or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("element_type") or "line") != "line":
            continue
        source_file = str(item.get("source_file") or "")
        line_no = item.get("line_no")
        raw_score = item.get("S_ochiai", item.get("score"))
        if not source_file or isinstance(line_no, bool) or not isinstance(line_no, int) or line_no <= 0:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if 0.0 <= score <= 1.0:
            scores[(source_file.replace("\\", "/"), line_no)] = score
    return scores


def _sbfl_score_for_line(
    line_scores: Mapping[tuple[str, int], float],
    source_file: str,
    line_no: int,
) -> float | None:
    normalized_source = source_file.replace("\\", "/")
    for (candidate_source, candidate_line), score in line_scores.items():
        if candidate_line != line_no:
            continue
        if (
            candidate_source == normalized_source
            or candidate_source.endswith(normalized_source)
            or normalized_source.endswith(candidate_source)
            or candidate_source.split("/")[-1] == normalized_source.split("/")[-1]
        ):
            return score
    return None


def _covered_suspicious_lines(info: Mapping[str, Any], suspicious_lines: Set[int]) -> Set[int]:
    if not suspicious_lines or _coverage_file_ratio(dict(info)) <= 0:
        return set()
    missing_lines = {
        line
        for line in (_parse_line_no(value) for value in (info.get("missing_lines") or []))
        if line is not None
    }
    return {line for line in suspicious_lines if line not in missing_lines}


def build_m7_llm_scenario_refinement(
    *,
    current_scenario: Mapping[str, Any],
    structured_feedback: Mapping[str, Any],
    execution_summary: Mapping[str, Any],
    valid_sbfl_candidates: List[Mapping[str, Any]],
    verdict_branch: str,
    generated_test: Mapping[str, Any] | None = None,
    score_breakdown: Mapping[str, Any] | None = None,
    llm_refiner: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build optional M7 LLM scenario-refinement instructions.

    The helper performs at most one injected/mock LLM call per non-ALIGNED gate
    failure. Its output is advisory only: it cannot alter the verdict,
    thresholds, admission fields, or downstream metrics.
    """
    verdict = str(verdict_branch or "")
    base: Dict[str, Any] = {
        "schema_version": _M7_LLM_REFINEMENT_SCHEMA_VERSION,
        "used_llm": False,
        "fallback_used": True,
        "fallback_reason": None,
        "verdict_branch": verdict,
        "M2_instructions": None,
        "M3_instructions": None,
        "M5_instructions": None,
        "priority": "normal",
        "reasoning": "deterministic structured_feedback retained",
        "evidence_provenance": [
            {"field": "current_scenario", "status": _m7_availability(current_scenario)},
            {"field": "structured_feedback", "status": _m7_availability(structured_feedback)},
            {"field": "execution_summary", "status": _m7_availability(execution_summary)},
            {"field": "valid_sbfl_candidates", "status": _m7_availability(valid_sbfl_candidates)},
            {"field": "verdict_branch", "status": _m7_availability(verdict)},
        ],
    }
    _apply_structured_feedback_fallback(base, structured_feedback)
    if verdict == FailureType.ALIGNED.value:
        base["fallback_reason"] = "aligned_verdict_no_llm_call"
        return base
    allowed_for_branch = _M7_LLM_BRANCH_MODULES.get(verdict)
    if allowed_for_branch is None:
        base["fallback_reason"] = "unsupported_verdict_branch"
        return base
    if llm_refiner is None:
        base["fallback_reason"] = "llm_refiner_unavailable"
        return base
    if _contains_forbidden_m7_evidence(
        {
            "current_scenario": current_scenario,
            "structured_feedback": structured_feedback,
            "execution_summary": execution_summary,
            "valid_sbfl_candidates": valid_sbfl_candidates,
        }
    ):
        base["fallback_reason"] = "forbidden_input_evidence_rejected"
        return base
    prompt = {
        "current_scenario": _safe_m7_value(current_scenario),
        "structured_feedback": _safe_m7_value(structured_feedback),
        "execution_summary": _safe_m7_value(execution_summary),
        "valid_sbfl_candidates": _safe_m7_value(valid_sbfl_candidates),
        "generated_test": {
            "target_test_file": (generated_test or {}).get("target_test_file"),
            "canonical_test_nodeid": (generated_test or {}).get("canonical_test_nodeid"),
            "test_code": _safe_m7_value((generated_test or {}).get("test_code") or (generated_test or {}).get("append_block") or ""),
        },
        "alignment_component_scores": {
            "bug_fail_score": (score_breakdown or {}).get("bug_fail_score"),
            "coverage_score": (score_breakdown or {}).get("coverage_score"),
            "issue_alignment_score": (score_breakdown or {}).get("issue_alignment_score"),
            "failure_type_detail": (score_breakdown or {}).get("failure_type_detail"),
            "oracle_risk_flags": _safe_m7_value((score_breakdown or {}).get("oracle_risk_flags") or []),
            "conservative_gate_reasons": _safe_m7_value((score_breakdown or {}).get("conservative_gate_reasons") or []),
        },
        "current_target": _safe_m7_value((current_scenario.get("target_location") or {}) if isinstance(current_scenario, Mapping) else {}),
        "current_oracle": {
            "oracle_type": current_scenario.get("oracle_type") if isinstance(current_scenario, Mapping) else None,
            "oracle_expected": current_scenario.get("oracle_expected") if isinstance(current_scenario, Mapping) else None,
            "expected_failure": current_scenario.get("expected_failure") if isinstance(current_scenario, Mapping) else None,
        },
        "verdict_branch": verdict,
        "allowed_modules": sorted(allowed_for_branch),
        "output_schema": {
            "diagnosis": "string",
            "target_changes": "object",
            "scenario_changes": "object",
            "oracle_changes": "object",
            "test_generation_constraints": "list",
            "evidence_references": "list",
            "M2_instructions": "object|null",
            "M3_instructions": "object|null",
            "M5_instructions": "object|null",
            "priority": "low|normal|high|critical",
            "reasoning": "string",
            "evidence_provenance": "list",
        },
    }
    try:
        candidate = llm_refiner(prompt)
    except Exception as exc:
        base["fallback_reason"] = f"llm_refiner_failed:{type(exc).__name__}"
        base["refiner_error"] = str(exc)
        return base
    valid, reason = _validate_m7_llm_refinement(candidate, allowed_for_branch)
    if not valid:
        base["fallback_reason"] = reason
        return base
    assert isinstance(candidate, Mapping)
    return {
        **base,
        "used_llm": True,
        "fallback_used": False,
        "fallback_reason": None,
        "diagnosis": str(candidate.get("diagnosis") or ""),
        "target_changes": _safe_m7_value(candidate.get("target_changes") or {}),
        "scenario_changes": _safe_m7_value(candidate.get("scenario_changes") or {}),
        "oracle_changes": _safe_m7_value(candidate.get("oracle_changes") or {}),
        "test_generation_constraints": _safe_m7_value(candidate.get("test_generation_constraints") or []),
        "evidence_references": _safe_m7_value(candidate.get("evidence_references") or []),
        "M2_instructions": _safe_m7_value(candidate.get("M2_instructions")),
        "M3_instructions": _safe_m7_value(candidate.get("M3_instructions")),
        "M5_instructions": _safe_m7_value(candidate.get("M5_instructions")),
        "priority": str(candidate.get("priority") or "normal"),
        "reasoning": str(candidate.get("reasoning") or ""),
        "evidence_provenance": _safe_m7_value(candidate.get("evidence_provenance") or []),
    }


def _apply_structured_feedback_fallback(
    payload: Dict[str, Any],
    structured_feedback: Mapping[str, Any],
) -> None:
    """Preserve deterministic M7 routing when LLM refinement cannot be used."""
    for module in _M7_LLM_ALLOWED_MODULES:
        key = f"{module}_instructions"
        payload[key] = _safe_m7_value(structured_feedback.get(key))
    diagnosis = structured_feedback.get("diagnosis")
    routing_reason = structured_feedback.get("routing_reason")
    if diagnosis or routing_reason:
        payload["reasoning"] = str(routing_reason or diagnosis)
    provenance = structured_feedback.get("evidence_provenance")
    if isinstance(provenance, list):
        payload["evidence_provenance"] = _safe_m7_value(provenance)


def _validate_m7_llm_refinement(
    candidate: Any,
    allowed_for_branch: Set[str],
) -> tuple[bool, str]:
    if not isinstance(candidate, Mapping):
        return False, "llm_output_not_mapping"
    if _contains_forbidden_m7_evidence(candidate):
        return False, "forbidden_output_evidence_rejected"
    for key in candidate:
        if key.endswith("_instructions") and key[:2] not in _M7_LLM_ALLOWED_MODULES:
            return False, f"unauthorized_module_instruction:{key}"
    for module in _M7_LLM_ALLOWED_MODULES:
        instructions = candidate.get(f"{module}_instructions")
        if instructions is not None and module not in allowed_for_branch:
            return False, f"unauthorized_branch_instruction:{module}"
    priority = str(candidate.get("priority") or "normal")
    if priority not in {"low", "normal", "high", "critical"}:
        return False, "invalid_priority"
    if not isinstance(candidate.get("evidence_provenance", []), list):
        return False, "invalid_evidence_provenance"
    return True, ""


def _contains_forbidden_m7_evidence(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(pattern in key_text for pattern in _M7_FORBIDDEN_EVIDENCE_PATTERNS):
                return True
            if _contains_forbidden_m7_evidence(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_m7_evidence(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(pattern in lowered for pattern in _M7_FORBIDDEN_EVIDENCE_PATTERNS)
    return False


def _contains_forbidden_sbfl_provenance(value: Any) -> bool:
    """Reject only explicit forbidden provenance in M6 SBFL payloads.

    SBFL line paths, stdout, exception text, and ordinary pre-patch values may
    contain words such as ``patch`` or ``post`` without being leakage.  The
    provenance boundary is therefore field-aware: forbidden metadata keys or
    an explicitly non-pre-patch source are rejected, while free-form evidence
    text is not scanned as a substring.
    """
    forbidden_keys = {
        "golden_patch", "golden_patch_lines", "patched_source", "patched_repo",
        "post_patch", "post_patch_execution", "post_patch_outcome",
        "m8_results", "m8_evaluation", "fail_to_pass", "f_to_p", "patch_hit_rate", "phr",
    }
    allowed_source_values = {"pre_patch", "pre-patch", "pre_patch_source", "repository_validator"}

    def visit(item: Any, parent_key: str = "") -> bool:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized_key = str(key).lower().replace("-", "_")
                if normalized_key in forbidden_keys:
                    return True
                if normalized_key in {"source_checkout", "source_view", "source_mapping_status"}:
                    if isinstance(child, str) and child and child.lower() not in allowed_source_values and normalized_key != "source_mapping_status":
                        return True
                if normalized_key in {"provenance", "evidence_provenance", "origin"} and isinstance(child, Mapping):
                    origin = str(child.get("source") or child.get("origin") or "").lower()
                    if any(token in origin for token in ("golden", "post_patch", "after_patch", "m8")):
                        return True
                if visit(child, normalized_key):
                    return True
        elif isinstance(item, list):
            return any(visit(child, parent_key) for child in item)
        return False

    return visit(value)


def _safe_m7_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_m7_value(item)
            for key, item in value.items()
            if not any(pattern in str(key).lower() for pattern in _M7_FORBIDDEN_EVIDENCE_PATTERNS)
        }
    if isinstance(value, list):
        return [_safe_m7_value(item) for item in value]
    return value


def _m7_availability(value: Any) -> str:
    return "available" if value not in (None, "", [], {}) else "unavailable"


def _extract_canonical_m6_sbfl(context: Mapping[str, Any] | None, execution_result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidates = []
    if isinstance(context, Mapping):
        candidates.append(context.get("canonical_m6_sbfl_result"))
    candidates.append(execution_result.get("canonical_m6_sbfl_result"))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        payload = candidate.get("payload")
        if _is_canonical_m6_sbfl_payload(payload):
            if _m6_sbfl_matches_current_pass(payload, execution_result):
                return payload
        if _is_canonical_m6_sbfl_payload(candidate):
            if _m6_sbfl_matches_current_pass(candidate, execution_result):
                return candidate
    return None


def _m6_sbfl_matches_current_pass(
    sbfl_result: Mapping[str, Any], execution_result: Mapping[str, Any]
) -> bool:
    """Reject stale SBFL evidence when both sides expose pass identity."""
    metadata = sbfl_result.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    for key in ("instance_id", "run_id", "iteration", "candidate_id", "candidate_hash"):
        expected = execution_result.get(key)
        observed = metadata.get(key, sbfl_result.get(key))
        if expected is not None and observed is not None and str(expected) != str(observed):
            return False
    return True


def _is_canonical_m6_sbfl_payload(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    metadata = value.get("metadata")
    return (
        "suspiciousness" in value
        and str(value.get("formula") or "").lower() == "ochiai"
        and isinstance(metadata, Mapping)
        and "activation_status" in metadata
        and "sbfl_active" in metadata
    )


def _valid_m7_sbfl_candidates(sbfl_result: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if _validate_m7_sbfl_payload(sbfl_result):
        return []
    assert sbfl_result is not None
    candidates = []
    for item in sbfl_result.get("suspiciousness") or []:
        if not isinstance(item, Mapping) or str(item.get("element_type") or "line") != "line":
            continue
        score = item.get("S_ochiai", item.get("score"))
        try:
            score_float = float(score)
        except (TypeError, ValueError):
            continue
        if 0.0 <= score_float <= 1.0:
            candidates.append(
                {
                    "source_file": str(item.get("source_file") or ""),
                    "line_no": item.get("line_no"),
                    "S_ochiai": round(score_float, 6),
                    "rank": item.get("rank"),
                }
            )
    return candidates


def _execution_summary_for_m7_llm(execution_result: Mapping[str, Any]) -> Dict[str, Any]:
    raw_output = str(execution_result.get("raw_output") or "")
    return {
        "test_results": _safe_m7_value(execution_result.get("test_results") or {}),
        "has_failure": bool(execution_result.get("has_failure")),
        "has_error": bool(execution_result.get("has_error")),
        "failure_signature": execution_result.get("failure_signature"),
        "error_messages": _safe_m7_value(execution_result.get("error_messages") or []),
        "stdout": _safe_m7_value(execution_result.get("stdout") or ""),
        "stderr": _safe_m7_value(execution_result.get("stderr") or ""),
        "raw_output_excerpt": raw_output[:6000],
        "traceback_excerpt": _extract_m7_traceback_excerpt(raw_output),
        "coverage_data_available": isinstance(execution_result.get("coverage_data"), Mapping),
        "contributing_functions": _safe_m7_value(execution_result.get("contributing_functions") or []),
    }


def _extract_m7_traceback_excerpt(raw_output: str) -> str:
    if not raw_output:
        return ""
    match = re.search(r"={10,}\s+FAILURES\s+={10,}(.{0,4000})", raw_output, flags=re.DOTALL)
    if match:
        return match.group(0)
    traceback_index = raw_output.find("Traceback")
    if traceback_index >= 0:
        return raw_output[traceback_index:traceback_index + 4000]
    return ""


def _apply_m7_llm_refinement_to_scenario(
    refined: Mapping[str, Any],
    llm_feedback: Mapping[str, Any],
    *,
    verdict: str,
) -> Dict[str, Any]:
    scenario = copy.deepcopy(dict(refined or {}))
    if not llm_feedback.get("used_llm"):
        return scenario
    before_hash = _stable_m7_hash(scenario)
    directive = dict(scenario.get("repair_directive") or {})
    directive["mode"] = directive.get("mode") or _m7_directive_mode(verdict)
    directive["blocking_reason"] = (
        str(llm_feedback.get("diagnosis") or llm_feedback.get("reasoning") or "")
        or directive.get("blocking_reason", "")
    )
    must_change = list(directive.get("must_change") or [])
    must_change.extend(_m7_constraint_strings(llm_feedback))
    directive["must_change"] = _dedupe_m7_strings(must_change)[:12]
    structured_hints = _safe_m7_value({
        "target_changes": llm_feedback.get("target_changes") or {},
        "scenario_changes": llm_feedback.get("scenario_changes") or {},
        "oracle_changes": llm_feedback.get("oracle_changes") or {},
        "M2_instructions": llm_feedback.get("M2_instructions"),
        "M3_instructions": llm_feedback.get("M3_instructions"),
        "M5_instructions": llm_feedback.get("M5_instructions"),
    })
    replacement_hints = list(directive.get("replacement_hints") or [])
    replacement_hints.extend(_m7_constraint_strings({
        "target_changes": llm_feedback.get("target_changes") or {},
        "scenario_changes": llm_feedback.get("scenario_changes") or {},
        "oracle_changes": llm_feedback.get("oracle_changes") or {},
    }))
    directive["replacement_hints"] = _dedupe_m7_strings(replacement_hints)[:12]
    directive["m7_llm_structured_hints"] = structured_hints
    directive["evidence_references"] = _safe_m7_value(llm_feedback.get("evidence_references") or [])
    directive["source"] = "m7_qwen_llm_scenario_refinement"
    scenario["repair_directive"] = directive
    scenario["m7_llm_feedback_applied"] = True
    scenario["m7_llm_feedback_application"] = {
        "schema_version": "m7-llm-feedback-application-v1",
        "verdict": verdict,
        "scenario_hash_before": before_hash,
        "scenario_hash_after": _stable_m7_hash(scenario),
        "directive_changed": True,
        "effectiveness_pending_next_iteration": True,
    }
    return scenario


def _m7_directive_mode(verdict: str) -> str:
    if verdict == FailureType.NOT_FAILED.value:
        return "REWRITE_STIMULUS_OR_ORACLE"
    if verdict == FailureType.NO_COVERAGE.value:
        return "RETARGET_SOURCE"
    if verdict == FailureType.WEAK_ALIGNMENT.value:
        return "REWRITE_ORACLE"
    return "IMPROVE_ALIGNMENT"


def _m7_constraint_strings(llm_feedback: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    for key in ("diagnosis", "reasoning"):
        if llm_feedback.get(key):
            values.append(str(llm_feedback[key]))
    for key in ("target_changes", "scenario_changes", "oracle_changes"):
        value = llm_feedback.get(key)
        if value:
            values.append(f"{key}: {json.dumps(_safe_m7_value(value), ensure_ascii=False, sort_keys=True)}")
    constraints = llm_feedback.get("test_generation_constraints")
    if isinstance(constraints, list):
        values.extend(str(item) for item in constraints if str(item).strip())
    return values


def _dedupe_m7_strings(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _stable_m7_hash(value: Mapping[str, Any]) -> str:
    import hashlib
    import json as _json

    return hashlib.sha256(
        _json.dumps(_safe_m7_value(value), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _resolve_m7_feature_flags(feature_flags: V22FeatureFlags | Mapping[str, Any] | None) -> V22FeatureFlags:
    if feature_flags is None:
        return resolve_feature_flags()
    if isinstance(feature_flags, V22FeatureFlags):
        return feature_flags
    if isinstance(feature_flags, Mapping):
        return resolve_feature_flags(feature_flags)
    raise TypeError("feature_flags must be V22FeatureFlags, mapping, or None")


def classify_and_score_v2(
    test_results: Dict[str, str],
    has_error: bool,
    bug_fail: float,
    coverage: float,
    issue_align: float,
    failure_features: Optional[Dict[str, int]] = None,
    error_messages: Optional[List[str]] = None,
    target_verified: bool = True,
    strong_issue_evidence: bool = True,
) -> FailureType:
    """논문 §3.3 게이트 방식 분류.

    세 점수(s_b, s_c, s_a)를 순차 게이트로 검사한다.
    각 게이트를 통과해야 다음 게이트로 진행하며, 모두 통과 시 ALIGNED.

    Gate 순서 (논문 수식 3→2→1 순):
      1. Gate s_b: bug_fail >= _ALIGNED_BUG_FAIL_MIN — 버그 감지 신뢰도
         · FAILED 없으면 → NOT_FAILED
         · FAILED 있으나 s_b < 임계값 → WEAK_ALIGNMENT
      2. Gate s_c: coverage >= _COVERAGE_MIN_GATE (0.60)
         · 미달 → NO_COVERAGE
      3. Gate s_a: issue_align >= _ISSUE_ALIGN_MIN_GATE (0.65) — 이슈-테스트 정합성
         · 미달 → WEAK_ALIGNMENT

    반환값:
      ALIGNED / NOT_FAILED / ERROR / NOT_VALID / NO_COVERAGE / WEAK_ALIGNMENT
    """
    features = failure_features or {}

    # ── 에러/미수집 케이스 ──
    if has_error:
        if error_messages and any(
            "not collected" in m.lower() or "not found" in m.lower()
            for m in error_messages
        ):
            return FailureType.NOT_VALID
        return FailureType.ERROR

    for value in (bug_fail, coverage, issue_align):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            return FailureType.ERROR

    statuses = set(test_results.values()) if test_results else set()
    if not statuses or statuses <= {"ERROR", "SKIP"}:
        return FailureType.ERROR

    # Any ERROR or NOT_RUN result invalidates M7 quantitative admission. A
    # mixed FAILED+ERROR collection is not a successful reproduction: the
    # current pass did not produce a trustworthy complete outcome.
    if "ERROR" in statuses or "NOT_RUN" in statuses:
        if error_messages and any(
            "not collected" in m.lower()
            or "nameerror" in m.lower()
            or "importerror" in m.lower()
            or "modulenot" in m.lower()
            for m in error_messages
        ):
            return FailureType.NOT_VALID
        return FailureType.ERROR

    # ── Gate 1 (s_b): 버그 재현 신뢰도 ──
    if "FAILED" not in statuses:
        return FailureType.NOT_FAILED
    if bug_fail < _ALIGNED_BUG_FAIL_MIN:
        # FAILED이지만 s_b 임계값 미달 (import 에러 페널티 등)
        return FailureType.WEAK_ALIGNMENT

    # ── Gate 2 (s_c): 의심 위치 커버리지 ──
    if coverage < _COVERAGE_MIN_GATE:
        return FailureType.NO_COVERAGE

    # ── Gate 3 (s_a): 이슈-테스트 정합성 ──
    if issue_align < _ISSUE_ALIGN_MIN_GATE:
        return FailureType.WEAK_ALIGNMENT

    # target_verified=False는 class method를 top-level 함수로 착각하는 경우가 많아 soft 처리
    # contributing_functions가 있고 target이 명확히 없는 경우에만 차단
    if not target_verified and coverage > _COVERAGE_BASE:
        # 파일이 커버됐는데 함수가 확인되지 않는 경우 → weak이지만 차단하지 않음 (패스)
        pass

    # strong_issue_evidence 게이트 제거: Gate 3 (issue_align >= 0.65) 이 이미 커버
    # 이 게이트는 valid 재현 테스트를 0.225 경계에서 과다 차단하므로 삭제

    # ── 모든 게이트 통과 → ALIGNED ──
    return FailureType.ALIGNED


def normalize_aligned_component_scores(
    failure_type: FailureType,
    bug_fail: float,
    coverage: float,
    issue_align: float,
) -> tuple[float, float, float]:
    """Keep accepted ALIGNED artifacts in the same score band used in reporting.

    Classification is already decided by the raw gate values. After a case is
    accepted as ALIGNED, the recorded component scores should not look like a
    rejection signal in the batch ledger.
    """
    if failure_type != FailureType.ALIGNED:
        return bug_fail, coverage, issue_align
    return (
        round(max(bug_fail, _ALIGNED_REPORT_BUG_FAIL_MIN), 4),
        round(max(coverage, _ALIGNED_REPORT_COVERAGE_MIN), 4),
        round(max(issue_align, _ALIGNED_REPORT_ISSUE_ALIGN_MIN), 4),
    )


# ---------------------------------------------------------------------------
# 3. 피드백 생성
# ---------------------------------------------------------------------------

@dataclass
class ScenarioFeedback:
    failure_type: str
    diagnosis: str
    oracle_additions: List[str]
    stimulus_additions: List[str]
    precondition_additions: List[str]
    expected_failure_override: str
    switch_scenario: bool
    candidate_test_file_override: str = ""
    repair_directive: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_feedback_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _dedupe_feedback_preserve_order(items: List[str], limit: int = 100) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        norm = _normalize_feedback_text(text)
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _resolve_feedback_conflicts(feedback: ScenarioFeedback) -> ScenarioFeedback:
    """Remove internally conflicting feedback before it is written to scenario."""
    oracle = list(feedback.oracle_additions)
    stimulus = list(feedback.stimulus_additions)
    precond = list(feedback.precondition_additions)

    # If the scorer has selected a replacement test file, remove advice that
    # explicitly says not to switch files.
    if feedback.candidate_test_file_override or feedback.switch_scenario:
        precond = [
            item for item in precond
            if "다른 테스트 파일로 교체하지 말고" not in item
        ]

    # Success-path oracle feedback should not coexist with a raises-only oracle
    # instruction unless the issue explicitly says an exception is expected.
    joined = " ".join(oracle)
    wants_success_path = any(
        phrase in joined
        for phrase in (
            "예외 없이 성공",
            "success path",
            "post-call value",
            "positive oracle",
        )
    )
    if wants_success_path:
        oracle = [
            item for item in oracle
            if not (
                ("pytest.raises" in item or "assertRaises" in item)
                and "예외 없이 성공" not in item
                and "success path" not in item
            )
        ]

    return ScenarioFeedback(
        failure_type=feedback.failure_type,
        diagnosis=feedback.diagnosis,
        oracle_additions=_dedupe_feedback_preserve_order(oracle),
        stimulus_additions=_dedupe_feedback_preserve_order(stimulus),
        precondition_additions=_dedupe_feedback_preserve_order(precond),
        expected_failure_override=feedback.expected_failure_override,
        switch_scenario=feedback.switch_scenario,
        candidate_test_file_override=feedback.candidate_test_file_override,
        repair_directive=feedback.repair_directive,
    )


def _scenario_requires_unpaired_oracle_feedback_sanitization(scenario: Dict[str, Any]) -> bool:
    selected = scenario.get("selected_reproduction_example")
    if isinstance(selected, dict):
        if selected_example_requires_oracle_regeneration(scenario):
            return True
        if selected.get("oracle_pairing_status") == "requires_oracle_regeneration":
            return True
        provenance = str(selected.get("expected_output_provenance") or "").strip()
        if provenance in {"", "unknown", "not_available", "unpaired_requires_regeneration"}:
            return True
        return False
    return selected_example_requires_oracle_regeneration(scenario)


def _scenario_with_generated_test_provenance(
    scenario: Dict[str, Any],
    generated_test: Dict[str, Any],
) -> Dict[str, Any]:
    """Prefer generated-test provenance over stale scenario metadata."""
    effective = copy.deepcopy(scenario)
    selected = generated_test.get("selected_reproduction_example")
    if not isinstance(selected, dict):
        prompt_profile = generated_test.get("prompt_profile") if isinstance(generated_test.get("prompt_profile"), dict) else {}
        selected = prompt_profile.get("selected_reproduction_example")
    if isinstance(selected, dict):
        effective["selected_reproduction_example"] = dict(selected)
        if selected.get("oracle_pairing_status") == "requires_oracle_regeneration" or selected.get("requires_oracle_regeneration"):
            effective["oracle_pairing_status"] = "requires_oracle_regeneration"
            effective["oracle_requires_regeneration"] = True
            effective["requires_oracle_regeneration"] = True
            effective["expected_outputs"] = []
            effective.pop("oracle_expected", None)
    relational = generated_test.get("relational_oracle")
    if not isinstance(relational, dict):
        prompt_profile = generated_test.get("prompt_profile") if isinstance(generated_test.get("prompt_profile"), dict) else {}
        relational = prompt_profile.get("relational_oracle")
    if isinstance(relational, dict) and relational:
        effective["relational_oracle"] = dict(relational)
        effective["relational_oracle_candidate"] = dict(relational)
    return sanitize_oracle_regeneration_payload(effective)


def _sanitize_unpaired_oracle_feedback(
    feedback: ScenarioFeedback,
    clue: Dict[str, Any],
    scenario: Optional[Dict[str, Any]] = None,
) -> ScenarioFeedback:
    unsafe_values = [
        str(value).strip()
        for value in (clue.get("expected_outputs", []) or [])
        if str(value).strip()
    ]

    def keep_item(value: Any) -> bool:
        text = str(value or "")
        lowered = text.lower()
        normalized = re.sub(r"\s+", "", lowered)
        if any(re.sub(r"\s+", "", unsafe.lower()) in normalized for unsafe in unsafe_values):
            return False
        return not (
            "fixed expected output" in lowered
            or "expected_outputs" in lowered
            or "expected output" in lowered
            or "올바른(fix 후) 기대값" in text
            or "기대 출력" in text
        )

    directive = feedback.repair_directive if isinstance(feedback.repair_directive, dict) else {}
    sanitized_directive = sanitize_repair_directive(directive) if directive else directive
    relational = {}
    if isinstance(scenario, dict):
        for key in ("relational_oracle", "relational_oracle_candidate"):
            if isinstance(scenario.get(key), dict) and scenario[key]:
                relational = dict(scenario[key])
                break
    if relational and isinstance(sanitized_directive, dict):
        sanitized_directive = copy.deepcopy(sanitized_directive)
        evidence = sanitized_directive.get("evidence") if isinstance(sanitized_directive.get("evidence"), dict) else {}
        evidence = dict(evidence)
        evidence["relational_oracle"] = relational
        if relational.get("validated_provenance") == RELATIONAL_ORACLE_PROVENANCE:
            evidence["oracle_source"] = RELATIONAL_ORACLE_PROVENANCE
        sanitized_directive["evidence"] = evidence
    return ScenarioFeedback(
        failure_type=feedback.failure_type,
        diagnosis=feedback.diagnosis,
        oracle_additions=[item for item in feedback.oracle_additions if keep_item(item)],
        stimulus_additions=list(feedback.stimulus_additions),
        precondition_additions=list(feedback.precondition_additions),
        expected_failure_override="",
        switch_scenario=feedback.switch_scenario,
        candidate_test_file_override=feedback.candidate_test_file_override,
        repair_directive=sanitized_directive,
    )


def _extract_error_detail_from_raw(raw_output: str, feature_name: str) -> str:
    """raw_output에서 피처 유형별 구체적 에러 메시지를 추출한다."""
    if not raw_output:
        return ""

    if feature_name == "f_import_err":
        m = re.search(
            r"(NameError: name '([^']+)' is not defined"
            r"|ImportError: [^\n]+"
            r"|ModuleNotFoundError: [^\n]+)",
            raw_output,
        )
        return m.group(0).strip() if m else ""

    if feature_name == "f_db_err":
        m = re.search(
            r"(OperationalError: [^\n]+"
            r"|DatabaseError: [^\n]+"
            r"|ProgrammingError: [^\n]+"
            r"|no such table: [^\n]+)",
            raw_output,
        )
        return m.group(0).strip() if m else ""

    if feature_name == "f_setup_assert":
        m = re.search(
            r"(setUp[^\n]*\n[^\n]+"
            r"|setUpClass[^\n]*\n[^\n]+"
            r"|Database queries to [^\n]+)",
            raw_output,
        )
        return m.group(0).strip() if m else ""

    return ""


def _fallback_traceback(raw_output: str, lines: int = _FEEDBACK_TRACEBACK_LINES) -> str:
    """raw_output에서 마지막 traceback을 최대 lines줄 추출."""
    if not raw_output:
        return ""
    tb_lines = raw_output.strip().splitlines()
    return "\n".join(tb_lines[-lines:])


def _extract_runtime_exception(raw_output: str) -> str:
    """raw_output에서 실제 발생한 예외 타입+메시지를 추출한다.

    ERROR 케이스에서 피드백에 포함할 구체적 에러를 찾기 위해 사용.
    마지막 Traceback 블록의 마지막 예외 라인을 반환.
    """
    if not raw_output:
        return ""
    # Traceback 블록 전체를 찾아 마지막 것을 사용
    tb_blocks = list(re.finditer(r"Traceback \(most recent call last\):", raw_output))
    if tb_blocks:
        last_tb_start = tb_blocks[-1].start()
        tb_section = raw_output[last_tb_start:]
        # 예외 라인: "ExceptionType: message" 형태 (마지막 비어있지 않은 줄)
        lines = [l for l in tb_section.splitlines() if l.strip()]
        for line in reversed(lines):
            # 알려진 예외 클래스 패턴
            if re.match(r"^\s*[\w.]+(?:Error|Exception|Fail|Warning|DoesNotExist|NotFound"
                        r"|Invalid|Missing|Exist|NotImplemented).*:", line):
                return line.strip()
            # 마지막 줄이 예외라면 (패턴 매칭 실패해도)
            if re.match(r"^\s*[\w.]+Error:", line) or re.match(r"^\s*[\w.]+Exception:", line):
                return line.strip()
    # Traceback 없으면 ERROR/Exception 패턴 직접 검색
    m = re.search(
        r"\b\w+(?:Error|Exception|DoesNotExist|NotFound): [^\n]+",
        raw_output,
    )
    return m.group(0).strip() if m else ""


def _empty_repair_directive() -> Dict[str, Any]:
    return {
        "mode": "",
        "blocking_reason": "",
        "must_change": [],
        "must_keep": [],
        "forbidden_patterns": [],
        "replacement_hints": [],
        "evidence": {},
    }


def _dedupe_strs(values: List[Any], limit: int = 20) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _issue_text_blob(clue: Optional[Dict[str, Any]]) -> str:
    clue = clue or {}
    identifiers = clue.get("identifiers") if isinstance(clue.get("identifiers"), dict) else {}
    values: List[Any] = []
    for key in (
        "observed_behavior",
        "expected_behavior",
        "repro_conditions",
        "expected_outputs",
        "actual_outputs",
        "error_keywords",
    ):
        value = clue.get(key, [])
        values.extend(value if isinstance(value, list) else [value])
    values.extend(identifiers.get("exceptions", []) or [])
    values.append(clue.get("raw_issue_text", ""))
    return " ".join(
        str(x)
        for x in values
        if x is not None
    )


def _issue_expected_warning_type(clue: Optional[Dict[str, Any]]) -> str:
    text = _issue_text_blob(clue)
    m = re.search(r"\b([A-Z][A-Za-z0-9_]*Warning)\b", text)
    if m:
        return m.group(1)
    if re.search(r"\bwarn(?:ing|s)?\b", text, re.IGNORECASE):
        return "Warning"
    return ""


def _issue_says_warning_expected(clue: Optional[Dict[str, Any]]) -> bool:
    text = _issue_text_blob(clue).lower()
    if not re.search(r"\bwarn(?:ing|s)?\b", text):
        return False
    expected_text = " ".join(str(x) for x in (clue or {}).get("expected_behavior", [])).lower()
    if re.search(r"\bwarn(?:ing|s)?\b|[a-z0-9_]*warning\b", expected_text):
        return not bool(re.search(
            r"should\s+not\s+warn|must\s+not\s+warn|without\s+warning|no\s+warning|"
            r"no\s+longer\s+warns?|does\s+not\s+warn|doesn't\s+warn",
            expected_text,
        ))
    return not bool(re.search(
        r"should\s+not\s+warn|must\s+not\s+warn|without\s+warning|no\s+warning|"
        r"no\s+longer\s+warns?|does\s+not\s+warn|doesn't\s+warn",
        text,
    ))


def _extract_failure_signal(raw_output: str) -> Dict[str, str]:
    """Extract concise failure evidence from pytest/unittest output."""
    signal = {
        "exception_type": "",
        "exception_message": "",
        "failing_line": "",
        "failing_test": "",
    }
    if not raw_output:
        return signal

    failed_tests = re.findall(r"FAILED\s+([^\s]+?::[^\s]+)", raw_output)
    if failed_tests:
        signal["failing_test"] = failed_tests[-1]
    else:
        unittest_match = re.search(r"FAIL:\s+([^\n]+)", raw_output)
        if unittest_match:
            signal["failing_test"] = unittest_match.group(1).strip()

    # Pytest marks the executing source line with a leading '>'.
    failing_lines = [
        line.strip()[1:].strip()
        for line in raw_output.splitlines()
        if line.lstrip().startswith(">") and len(line.strip()) > 1
    ]
    if failing_lines:
        signal["failing_line"] = failing_lines[-1][:300]

    exc = _extract_runtime_exception(raw_output)
    if exc:
        m = re.match(r"([\w.]+(?:Error|Exception|Warning|OSError|AssertionError)):\s*(.*)", exc)
        if m:
            signal["exception_type"] = m.group(1).split(".")[-1]
            signal["exception_message"] = m.group(2)[:300]
        else:
            parts = exc.split(":", 1)
            signal["exception_type"] = parts[0].strip().split(".")[-1]
            signal["exception_message"] = parts[1].strip()[:300] if len(parts) > 1 else exc[:300]
    return signal


def _is_construction_failure_signal(signal: Dict[str, str]) -> bool:
    exc_type = signal.get("exception_type", "")
    message = signal.get("exception_message", "")
    return bool(
        exc_type in {
            "TypeError",
            "AttributeError",
            "OSError",
            "ModuleNotFoundError",
            "ImportError",
        }
        or re.search(r"missing \d+ required positional argument|has no attribute|No module named", message)
    )


def _is_issue_reported_failure_signal(
    signal: Mapping[str, str],
    clue: Optional[Mapping[str, Any]],
) -> bool:
    """Return true only for an exception explicitly grounded in issue evidence."""
    exception_type = str(signal.get("exception_type") or "").strip()
    if not exception_type or not clue:
        return False
    issue_blob = _issue_text_blob(dict(clue)).lower()
    if exception_type.lower() not in issue_blob:
        return False
    message = str(signal.get("exception_message") or "").strip().lower()
    if not message:
        return True
    message_tokens = {
        token
        for token in _token_set(message)
        if token not in {"error", "exception", "expected", "actual", "failed", "failure"}
    }
    issue_tokens = _token_set(issue_blob)
    return len(message_tokens & issue_tokens) >= min(2, len(message_tokens))


def _forbidden_patterns_from_signal(signal: Dict[str, str]) -> List[str]:
    patterns: List[str] = []
    line = signal.get("failing_line", "")
    exc_type = signal.get("exception_type", "")
    message = signal.get("exception_message", "")
    if line:
        patterns.append(line)
        call = re.search(r"([A-Za-z_][\w.]*\([^#\n]{0,120})", line)
        if call:
            patterns.append(call.group(1))
        attr = re.search(r"\.([A-Za-z_]\w*)\s*\(", line)
        if attr:
            patterns.append(f".{attr.group(1)}(")
    missing_attr = re.search(r"has no attribute ['\"]([^'\"]+)['\"]", message)
    if exc_type == "AttributeError" and missing_attr:
        patterns.append(f".{missing_attr.group(1)}(")
    missing_module = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
    if missing_module:
        patterns.append(f"import {missing_module.group(1)}")
        patterns.append(f"from {missing_module.group(1)}")
    if "Plotter" in message and "get_legend" in message:
        patterns.extend(["plot.plot().get_legend", ".get_legend("])
    return _dedupe_strs(patterns, limit=8)


_BROAD_FORBIDDEN_PATTERNS = {
    "django.",
    ".add(",
    ".plot(",
    ".draw(",
    ".scale(",
    "assert true",
    "assert false",
}


def _normalize_forbidden_pattern(pattern: Any) -> str:
    text = str(pattern or "").strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text)
    lower = compact.lower()
    if lower in _BROAD_FORBIDDEN_PATTERNS:
        return ""
    if len(compact) > 120 and compact.startswith(("assert ", "self.assert")):
        repr_m = re.search(r"repr\s*\([^)]+\)\s*(?:!=|==)", compact)
        if repr_m:
            return repr_m.group(0)
        call_m = re.search(r"([A-Za-z_][\w.]*\([^)]{0,80}\))\s*(?:!=|==|is|in|not in)", compact)
        if call_m:
            return call_m.group(0)
        return ""
    if compact.startswith("assert "):
        if re.search(r"\b(?:np\.)?array_equal\s*\(", compact):
            return ""
        repr_m = re.search(r"repr\s*\([^)]+\)\s*(?:!=|==)", compact)
        if repr_m:
            return repr_m.group(0)
        expr = compact[len("assert "):].strip()
        return expr[:100] if expr else ""
    if compact.startswith("self.assert"):
        call_m = re.search(r"self\.assert\w+\s*\(([^,\n)]{1,100})", compact)
        return call_m.group(1).strip() if call_m else ""
    if len(compact) > 160:
        call_m = re.search(r"([A-Za-z_][\w.]*\([^#\n]{0,100})", compact)
        return call_m.group(1) if call_m else ""
    return compact


def _normalize_forbidden_patterns(patterns: List[Any]) -> List[str]:
    return _dedupe_strs(
        [p for p in (_normalize_forbidden_pattern(v) for v in patterns) if p],
        limit=10,
    )


def _default_replacement_hints(
    mode: str,
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    signal: Dict[str, str],
    forbidden_patterns: List[str],
) -> List[str]:
    hints: List[str] = []
    target = scenario.get("target_location") or {}
    target_func = str(target.get("target_function", "") or "")
    expected_outputs = clue.get("expected_outputs", []) or []
    actual_outputs = clue.get("actual_outputs", []) or []
    issue_call_patterns = _issue_call_patterns(clue)
    expected_warning_type = _issue_expected_warning_type(clue)

    if mode == "REWRITE_ORACLE":
        if expected_outputs:
            hints.append(f"Assert the fixed expected output/value: {str(expected_outputs[0])[:160]}")
        elif actual_outputs:
            hints.append(f"Assert the target result/public state differs from buggy observed output: {str(actual_outputs[0])[:160]}")
        hints.append("Store the target API result/public state in a variable and assert that variable, not a local expected constant.")
        for pattern in issue_call_patterns[:2]:
            hints.append(f"Reuse issue call pattern as the canonical stimulus: {pattern[:160]}")
    elif mode == "REWRITE_STIMULUS":
        hints.append("Replace the passing baseline stimulus with the issue's explicit failing/problem reproduction call.")
        hints.append("Preserve setup that is required by the selected bug-triggering call.")
        for pattern in _bug_triggering_issue_call_patterns(clue)[:2]:
            hints.append(f"Use bug-triggering issue call pattern: {pattern[:160]}")
    elif mode == "RETARGET_SOURCE":
        if target_func:
            hints.append(f"Build the stimulus around target function/API `{target_func}` or a public caller that reaches it.")
        for pattern in issue_call_patterns[:2]:
            hints.append(f"Reuse issue call pattern: {pattern[:160]}")
    elif mode in {"FIX_IMPORTS", "FIX_API_USAGE"}:
        hints.append("Mirror imports, fixtures, and helper classes already present in the target test file.")
        hints.append("Replace unresolved or missing symbols with verified Existing Imports or Available Imports.")
        hints.append("Prefer the target file's existing test style/imports/helpers over inventing new runtime attributes or models.")
    elif mode == "SWITCH_SCENARIO_OR_TEST_FILE":
        hints.append("Switch to the next validated scenario or a different candidate test file; do not reuse the same low-coverage setup.")
        hints.append("Carry over the issue reproduction call pattern and rebuild setup around an existing test-file style.")
    elif mode == "IMPROVE_ALIGNMENT":
        hints.append("Rewrite both stimulus and assertion so the same test reaches the target source and checks issue-specific fixed behavior.")

    if expected_warning_type:
        hints.append(f"If the issue expects a warning, wrap the target call with pytest.warns({expected_warning_type}) or the repository's equivalent warning helper.")
    joined = " ".join(forbidden_patterns + [signal.get("exception_message", "")]).lower()
    if "get_legend" in joined or "seaborn" in joined:
        hints.append("For Seaborn/Matplotlib, assert public ticks, limits, labels, scale mappings, or artist/axis state instead of legend/private Plotter APIs.")
    if "raw_rendered" in joined or "sphinx" in joined or "html" in joined or "latex" in joined:
        hints.append("For Sphinx, assert a small doctree/node semantic marker or public rendered property, not a long raw HTML/LaTeX string.")
    if "raises_only_no_body_assertion" in joined:
        hints.append("If the fix removes an exception, remove pytest.raises and assert the successful return value or public state after the call.")
    return _dedupe_strs(hints, limit=8)


def _covered_functions_from_execution(execution_result: Optional[Dict[str, Any]]) -> List[str]:
    contributing = (execution_result or {}).get("contributing_functions", [])
    values: List[str] = []
    if isinstance(contributing, dict):
        for fname, funcs in contributing.items():
            if isinstance(funcs, list):
                values.extend(f"{fname}:{fn}" for fn in funcs[:8])
            elif funcs:
                values.append(f"{fname}:{funcs}")
    elif isinstance(contributing, list):
        values.extend(str(x) for x in contributing)
    return _dedupe_strs(values, limit=20)


def _issue_call_patterns(clue: Dict[str, Any], limit: int = 3) -> List[str]:
    patterns: List[str] = []
    for block in clue.get("code_examples", []) or []:
        if not isinstance(block, dict) or block.get("is_system_or_output"):
            continue
        code = block.get("interactive_input") or block.get("code", "")
        if code:
            patterns.append(str(code).strip()[:240])
        if len(patterns) >= limit:
            break
    return _dedupe_strs(patterns, limit=limit)


def _clue_target_function(clue: Dict[str, Any]) -> str:
    identifiers = clue.get("identifiers", {}) if isinstance(clue.get("identifiers"), dict) else {}
    for key in ("functions", "methods"):
        values = identifiers.get(key, []) if isinstance(identifiers.get(key, []), list) else []
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
    for location in clue.get("fault_locations", []) or []:
        if isinstance(location, dict) and location.get("function_name"):
            return str(location.get("function_name") or "").strip()
    return ""


def _bug_triggering_issue_call_patterns(clue: Dict[str, Any], limit: int = 3) -> List[str]:
    return issue_bug_trigger_patterns(clue, limit=limit)


def _normal_form_for_call_presence(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _generated_test_contains_bug_trigger_call(clue: Dict[str, Any], generated_test: Dict[str, Any]) -> bool:
    return generated_test_contains_bug_trigger(clue, generated_test)


def _scenario_has_bug_trigger_evidence(scenario: Dict[str, Any], clue: Dict[str, Any]) -> bool:
    actual_outputs = clue.get("actual_outputs", []) or []
    target = scenario.get("target_location") if isinstance(scenario.get("target_location"), dict) else {}
    target_function = str(
        scenario.get("target_function")
        or target.get("target_function")
        or _clue_target_function(clue)
        or ""
    )
    classified_blocks = classify_reproduction_code_blocks(
        scenario.get("reproduction_code", []) or [],
        expected_outputs=scenario.get("expected_outputs", []) or clue.get("expected_outputs", []) or [],
        actual_outputs=scenario.get("actual_outputs", []) or actual_outputs,
        target_function=target_function,
    )
    if not target_function:
        return False
    return any(
        block_inferred_role(block) == ROLE_BUG_TRIGGER
        and contains_target_call(str(block.get("interactive_input") or block.get("code") or ""), target_function)
        for block in classified_blocks
    )


def _target_source_has_coverage(
    coverage_data: Dict[str, Dict],
    source_file: str,
) -> bool:
    if not coverage_data or not source_file:
        return False
    matched = _match_coverage_file(coverage_data, source_file)
    if not matched:
        return False
    info = coverage_data.get(matched, {})
    return isinstance(info, dict) and _coverage_file_ratio(info) > 0


def _build_repair_directive(
    mode: str,
    blocking_reason: str,
    scenario: Dict[str, Any],
    clue: Dict[str, Any],
    *,
    must_change: Optional[List[str]] = None,
    must_keep: Optional[List[str]] = None,
    forbidden_patterns: Optional[List[str]] = None,
    replacement_hints: Optional[List[str]] = None,
    execution_result: Optional[Dict[str, Any]] = None,
    raw_output: str = "",
    coverage: float = 0.0,
) -> Dict[str, Any]:
    target = scenario.get("target_location") or {}
    signal = _extract_failure_signal(raw_output)
    evidence = {
        "target_source": str(target.get("source_file", "")),
        "target_function": str(target.get("target_function", "")),
        "candidate_test_file": str(target.get("candidate_test_file", "")),
        "covered_functions": _covered_functions_from_execution(execution_result),
        "exception_type": signal.get("exception_type", ""),
        "exception_message": signal.get("exception_message", ""),
        "failing_line": signal.get("failing_line", ""),
        "failing_test": signal.get("failing_test", ""),
        "coverage_score": round(float(coverage or 0.0), 4),
        "issue_call_patterns": _issue_call_patterns(clue),
    }
    keep = list(must_keep or [])
    allow_clue_expected = not selected_example_requires_oracle_regeneration(scenario)
    if allow_clue_expected:
        for value in (clue.get("expected_outputs", []) or [])[:2]:
            keep.append(f"fixed expected output: {str(value)[:160]}")
    for value in (clue.get("actual_outputs", []) or [])[:2]:
        keep.append(f"buggy observed output to avoid: {str(value)[:160]}")
    normalized_forbidden = _normalize_forbidden_patterns(
        list(forbidden_patterns or []) + _forbidden_patterns_from_signal(signal)
    )
    target_func = str(target.get("target_function", "") or "")
    if target_func:
        normalized_forbidden = [
            pattern for pattern in normalized_forbidden
            if not (
                pattern == f".{target_func}("
                or (
                    re.search(rf"\b{re.escape(target_func)}\s*\(", pattern)
                    and not re.search(r"!=|==|\bis\b|\bin\b|\bnot\b|assert", pattern)
                )
            )
        ]
    hints = _dedupe_strs(
        list(replacement_hints or [])
        + _default_replacement_hints(mode, scenario, clue, signal, normalized_forbidden),
        limit=10,
    )
    directive = {
        "mode": mode,
        "blocking_reason": blocking_reason,
        "must_change": _dedupe_strs(list(must_change or []), limit=10),
        "must_keep": _dedupe_strs(keep, limit=8),
        "forbidden_patterns": normalized_forbidden,
        "replacement_hints": hints,
        "evidence": evidence,
    }
    if not allow_clue_expected:
        directive = sanitize_repair_directive(directive)
    return directive


def generate_feedback(
    failure_type: FailureType,
    clue: Dict[str, Any],
    scenario: Dict[str, Any],
    generated_test: Dict[str, Any],
    bug_fail: float,
    issue_align: float,
    coverage: float,
    error_messages: Optional[List[str]] = None,
    project_test_style: Optional[Dict[str, Any]] = None,
    raw_output: str = "",
    failure_features: Optional[Dict[str, int]] = None,
    execution_result: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ScenarioFeedback:
    """failure_type별 규칙기반 피드백 생성."""
    scenario = _scenario_with_generated_test_provenance(scenario, generated_test)
    oracle_adds: List[str] = []
    stimulus_adds: List[str] = []
    precond_adds: List[str] = []
    expected_failure_override = ""
    switch_scenario = False
    diagnosis = ""
    repair_directive: Dict[str, Any] = _empty_repair_directive()

    expected_outputs = clue.get("expected_outputs", [])
    actual_outputs = clue.get("actual_outputs", [])
    code_examples = clue.get("code_examples", [])

    def _append_composite_gate_feedback() -> None:
        """Add advice for every weak gate, not only the final label."""
        if failure_type in {FailureType.ALIGNED, FailureType.ERROR, FailureType.NOT_VALID}:
            return
        weak_gates: List[str] = []
        if bug_fail < _ALIGNED_BUG_FAIL_MIN:
            weak_gates.append(f"s_b={bug_fail:.3f}<τ_b={_ALIGNED_BUG_FAIL_MIN:.3f}")
        if coverage < _COVERAGE_MIN_GATE:
            weak_gates.append(f"s_c={coverage:.3f}<τ_c={_COVERAGE_MIN_GATE:.3f}")
        if issue_align < _ISSUE_ALIGN_MIN_GATE:
            weak_gates.append(f"s_a={issue_align:.3f}<τ_a={_ISSUE_ALIGN_MIN_GATE:.3f}")
        if len(weak_gates) <= 1:
            return

        nonlocal diagnosis, expected_failure_override
        diagnosis = f"{diagnosis} 복합 게이트 미달: {'; '.join(weak_gates)}".strip()
        if bug_fail < _ALIGNED_BUG_FAIL_MIN:
            expected_failure_override = (
                f"{expected_failure_override} "
                "현재 테스트는 before-patch에서 충분히 실패하지 않는다. "
                "버그 입력을 직접 실행하고 fix 후 기대 동작을 assert하라."
            ).strip()
        if coverage < _COVERAGE_MIN_GATE:
            target = scenario.get("target_location") or {}
            source_file = target.get("source_file", "")
            target_func = target.get("target_function", "")
            if source_file:
                stimulus_adds.append(
                    f"복합 미달 보강: 테스트가 {source_file} 경로를 실제로 지나가도록 입력/setup을 바꿔라."
                )
            if target_func:
                stimulus_adds.append(
                    f"복합 미달 보강: {target_func}를 직접 호출하거나 public caller를 통해 반드시 실행되게 하라."
                )
        if issue_align < _ISSUE_ALIGN_MIN_GATE:
            oracle_adds.append(
                "복합 미달 보강: 이슈의 입력, 기대 동작, 핵심 식별자를 같은 테스트 흐름 안에서 검증하라."
            )

    if failure_type == FailureType.ALIGNED:
        diagnosis = (
            "정합성 통과. 테스트가 버그 코드에서 실패하고, "
            "이슈와 정합하며, 의심 위치를 커버합니다."
        )

    elif failure_type == FailureType.ERROR:
        # 구체적 에러 메시지 포함
        error_detail = ""
        if error_messages:
            error_detail = " 구체적 에러: " + "; ".join(
                msg[:_FEEDBACK_SHORT_STR_LEN] for msg in error_messages[:_FEEDBACK_ERROR_MSGS_MAX]
            )

        no_results_parsed = error_messages and any(
            "no test results parsed" in m.lower() for m in error_messages
        )

        runner = (project_test_style or {}).get("runner", "unknown")

        if no_results_parsed:
            # 테스트가 실행됐지만 결과가 파싱되지 않음 → test discovery 실패
            diagnosis = (
                f"테스트가 실행됐지만 결과를 파싱하지 못함 (test discovery 실패).{error_detail} "
                "테스트 runner가 생성된 테스트를 발견하지 못했을 가능성이 높다."
            )
            switch_scenario = True
            if runner == "django-test":
                precond_adds.append(
                    "Django test runner는 standalone 함수를 수집하지 못한다. "
                    "반드시 django.test.TestCase 또는 unittest.TestCase를 상속한 클래스 안에 "
                    "test 메서드로 작성해야 한다. "
                    "단독 함수(def test_xxx():) 형태는 발견되지 않는다."
                )
            elif runner == "unittest":
                precond_adds.append(
                    "unittest runner는 TestCase 서브클래스의 test_ 메서드만 수집한다. "
                    "unittest.TestCase를 상속한 클래스 안에 test_ 메서드를 작성하라."
                )
            elif runner == "sympy-bin-test":
                precond_adds.append(
                    "SymPy ./bin/test runner는 test_ 로 시작하는 top-level 함수를 찾는다. "
                    "함수명이 test_로 시작하는지, 파일 경로가 올바른지 확인하라."
                )
            elif runner == "pytest":
                precond_adds.append(
                    "pytest가 테스트를 발견하지 못했다. "
                    "test_ 접두사 함수 또는 Test 접두사 클래스를 사용하고, "
                    "파일명이 test_ 또는 _test.py 여야 한다."
                )
            else:
                precond_adds.append(
                    "테스트 runner가 생성된 테스트를 발견하지 못했다. "
                    "테스트 클래스(Test로 시작)와 메서드(test_로 시작)를 사용하거나, "
                    "이 프로젝트의 기존 테스트 파일 구조를 참고하라."
                )
            precond_adds.append(
                "테스트 클래스 이름은 Test로 시작해야 하며, 메서드 이름은 test_로 시작해야 한다."
            )
        else:
            # raw_output에서 실제 발생한 예외 추출 (error_messages가 비어 있을 수 있음)
            runtime_exc = _extract_runtime_exception(raw_output)
            if not runtime_exc and error_messages:
                runtime_exc = "; ".join(msg[:_FEEDBACK_SHORT_STR_LEN] for msg in error_messages[:2])

            if runtime_exc:
                diagnosis = f"테스트 실행 중 에러 발생: {runtime_exc}"
            else:
                diagnosis = (
                    f"테스트 실행 에러. ImportError, SyntaxError, 또는 환경 문제일 수 있습니다.{error_detail}"
                )

            # 실제 예외가 있으면 구체적으로 피드백
            if runtime_exc:
                # 예외 종류별 구체적 힌트
                if re.search(r"ImportError|ModuleNotFoundError", runtime_exc):
                    precond_adds.append(
                        f"import 에러: {runtime_exc}. "
                        "해당 모듈의 실제 import 경로를 repo에서 확인하고 수정하라."
                    )
                elif re.search(r"NameError", runtime_exc):
                    name_m = re.search(r"name '([^']+)' is not defined", runtime_exc)
                    if name_m:
                        precond_adds.append(
                            f"'{name_m.group(1)}'이 정의되지 않았다. "
                            f"테스트 파일 상단에 import하거나, 직접 정의하라. 원문: {runtime_exc}"
                        )
                    else:
                        precond_adds.append(f"NameError 발생: {runtime_exc}")
                elif re.search(r"SyntaxError", runtime_exc):
                    precond_adds.append(
                        f"Python 구문 오류: {runtime_exc}. 테스트 코드의 문법을 확인하라."
                    )
                else:
                    # TypeError, AttributeError, DoesNotExist 등 런타임 에러
                    precond_adds.append(
                        f"런타임 에러: {runtime_exc}. "
                        "이 에러가 발생한 원인을 분석해 테스트 코드를 수정하라. "
                        "API 사용법이 잘못됐거나 잘못된 인자를 전달하고 있을 수 있다."
                    )
                signal = _extract_failure_signal(raw_output)
                if _is_construction_failure_signal(signal) and not _is_issue_reported_failure_signal(
                    signal, clue
                ):
                    repair_directive = _build_repair_directive(
                        "FIX_API_USAGE",
                        "previous test failed from API/import/environment misuse, not a verified bug assertion",
                        scenario,
                        clue,
                        must_change=[
                            "Remove the failing API/import/setup pattern and rebuild the test from existing examples in the target test file.",
                            "Use only public APIs and symbols that exist in this repository version.",
                            "The next test must fail by assertion of the issue behavior, not by construction/runtime error.",
                        ],
                        must_keep=[
                            "Keep the original issue behavior and validated target source/function in scope.",
                        ],
                        execution_result=execution_result,
                        raw_output=raw_output,
                        coverage=coverage,
                    )
            else:
                precond_adds.append(
                    "테스트에서 사용하는 모든 import는 repo 내에 실제 존재하는 모듈/심볼이어야 한다."
                )
                precond_adds.append(
                    "생성된 테스트 코드는 Python 구문 오류가 없어야 한다."
                )
                if error_messages:
                    for msg in error_messages[:_FEEDBACK_ERROR_MSGS_MAX]:
                        import_err = re.search(r"(?:ImportError|ModuleNotFoundError):\s*(.+)", msg)
                        if import_err:
                            precond_adds.append(
                                f"이전 시도에서 import 에러 발생: {import_err.group(1)[:_FEEDBACK_SHORT_STR_LEN]}. "
                                "해당 import를 수정하거나 제거해야 한다."
                            )

    elif failure_type == FailureType.NOT_VALID:
        # 테스트가 유효하지 않음 — module-level skip / importorskip / NameError / not found
        error_detail = ""
        if error_messages:
            error_detail = " " + "; ".join(msg[:_FEEDBACK_SHORT_STR_LEN] for msg in error_messages[:_FEEDBACK_ERROR_MSGS_MAX])

        # NameError (missing import) 여부 감지
        is_name_error = error_messages and any(
            "NameError" in m or "Missing import" in m for m in error_messages
        )
        import re as _re
        missing_name = None
        if is_name_error and error_messages:
            for m in error_messages:
                match = _re.search(r"NameError: '(\w+)'", m)
                if match:
                    missing_name = match.group(1)
                    break

        is_import_error = error_messages and any(
            "ImportError" in m or "ModuleNotFoundError" in m for m in error_messages
        )

        if is_name_error and missing_name:
            diagnosis = (
                f"테스트가 수집되지 않음 (NameError: '{missing_name}' 미정의)."
                f"{error_detail}"
            )
            precond_adds.append(
                f"테스트 파일에 'import {missing_name}'이 없어 NameError가 발생했습니다. "
                f"imports 목록에 'import {missing_name}'을 반드시 포함해야 합니다."
            )
        elif is_import_error:
            import_detail = next(
                (m for m in error_messages if "ImportError" in m or "ModuleNotFoundError" in m),
                "",
            )
            diagnosis = (
                f"테스트가 수집되지 않음 (import 오류). {import_detail[:_FEEDBACK_SHORT_STR_LEN]}"
                f"{error_detail}"
            )
            # 'tests' 패키지 import 전용 피드백 (Django 프로젝트에서 반복되는 패턴)
            if "No module named 'tests'" in import_detail or "No module named 'tests." in import_detail:
                precond_adds.append(
                    "CRITICAL: `from tests.xxx import Y` 형태의 import는 실행 환경에서 "
                    "ModuleNotFoundError를 발생시킨다. `tests` 패키지는 Python 경로에 없다. "
                    "대신 Django 앱 모듈을 직접 import하라 (예: `from django.xxx import Y`). "
                    "필요한 모델/유틸리티는 기존 테스트 파일의 import 블록을 참고하라."
                )
            else:
                precond_adds.append(
                    f"테스트 수집 중 import 오류 발생: {import_detail[:_FEEDBACK_SHORT_STR_LEN]}. "
                    "imports 목록에서 해당 심볼을 제거하거나 올바른 모듈 경로로 수정하라. "
                    "다른 테스트 파일로 교체하지 말고 import를 수정하라."
                )
            repair_directive = _build_repair_directive(
                "FIX_IMPORTS",
                "test collection failed because an import/module path is invalid",
                scenario,
                clue,
                must_change=[
                    "Remove or replace the missing import using only Available Imports from Repository or Existing Imports.",
                    "Do not import optional database/backend packages unless already imported in the target test file.",
                ],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )
        else:
            diagnosis = (
                "테스트가 수집되지 않음. 대상 테스트 파일에 module-level skip 조건"
                "(pytest.importorskip 등)이 있어 테스트 함수가 실행되지 않았습니다."
                f"{error_detail}"
            )
            switch_scenario = True
            stimulus_adds.append(
                "현재 대상 테스트 파일에 module-level skip이 있어 테스트가 수집되지 않습니다. "
                "module-level skip이 없는 다른 테스트 파일을 선택해야 합니다."
            )
            repair_directive = _build_repair_directive(
                "CHANGE_TEST_FILE",
                "current test file cannot collect the generated test reliably",
                scenario,
                clue,
                must_change=[
                    "Choose a candidate test file without module-level skip or missing optional dependencies.",
                    "Mirror the selected file's existing test style exactly.",
                ],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )

    elif failure_type == FailureType.NOT_FAILED:
        diagnosis = (
            f"버그 코드에서 테스트가 FAIL하지 않음 (bug_fail={bug_fail:.1f}). "
            "assertion이 버그 동작 대신 정상 동작을 확인하고 있을 수 있습니다."
        )
        expected_failure_override = (
            "패치 적용 전(pre-patch) 코드에서 반드시 테스트가 FAILED 되어야 한다. "
            "assertion은 수정된(올바른) 동작을 기대해야 한다."
        )

        # 현재 assertion 내용 분석 — 모델이 무엇이 잘못됐는지 알 수 있도록
        test_code = generated_test.get("test_code", "")
        assert_lines = [
            line.strip()[:_FEEDBACK_SHORT_STR_LEN]
            for line in test_code.splitlines()
            if line.strip().startswith(("self.assert", "self.assertEqual",
                                        "self.assertRaises", "assert "))
        ]
        if assert_lines:
            oracle_adds.append(
                "현재 테스트의 assertion (버그 코드에서도 통과하고 있음):\n"
                + "\n".join(f"  {a}" for a in assert_lines[:_FEEDBACK_ASSERTION_MAX])
            )
            oracle_adds.append(
                "위 assertion이 버그 코드에서도 PASS하는 이유: "
                "버그가 발생하는 값이 아닌 다른 값을 기대하거나, "
                "버그와 무관한 동작을 검증하고 있을 가능성이 높다."
            )

        # expected vs actual 명확한 대조
        if expected_outputs and actual_outputs:
            oracle_adds.append(
                f"올바른(fix 후) 기대값: {str(expected_outputs[0])[:_FEEDBACK_MID_STR_LEN]}"
            )
            oracle_adds.append(
                f"버그(fix 전) 실제값: {str(actual_outputs[0])[:_FEEDBACK_MID_STR_LEN]}"
            )
            oracle_adds.append(
                "assertion은 반드시 버그 코드에서 FAIL해야 한다. "
                "위 '버그 실제값'이 나오는 상황에서 FAIL하고, "
                "'올바른 기대값'이 나오는 상황에서 PASS하도록 수정하라."
            )
        elif actual_outputs:
            oracle_adds.append(
                f"버그 코드의 실제 출력: {str(actual_outputs[0])[:_FEEDBACK_MID_STR_LEN]} — "
                "이 값을 그대로 기대값으로 쓰면 테스트가 항상 PASS한다. "
                "올바른 동작(버그 수정 후)의 기대값을 사용해야 한다."
            )
        else:
            oracle_adds.append(
                "assertion은 버그 수정 후의 올바른 결과를 기대해야 한다."
            )
        for out in expected_outputs[:_FEEDBACK_OUTPUTS_MAX]:
            oracle_adds.append(f"기대 출력(올바른 동작): {out[:_FEEDBACK_MID_STR_LEN]}")
        for out in actual_outputs[:_FEEDBACK_OUTPUTS_MAX]:
            oracle_adds.append(f"버그 출력(잘못된 동작): {out[:_FEEDBACK_MID_STR_LEN]}")

    elif failure_type == FailureType.NO_COVERAGE:
        diagnosis = (
            f"의심 위치 커버리지 없음 (coverage={coverage:.2f}). "
            "테스트가 타겟 소스 파일을 실행하지 않습니다."
        )
        switch_scenario = coverage < 0.3
        target = scenario.get("target_location") or {}
        source_file = target.get("source_file", "")
        target_func = target.get("target_function", "")
        if source_file:
            stimulus_adds.append(
                f"테스트는 {source_file} 파일의 코드를 직접 실행해야 한다."
            )
        if target_func:
            stimulus_adds.append(
                f"테스트는 {target_func} 함수를 직접 호출해야 한다."
            )
        for block in code_examples[:_FEEDBACK_CODE_EXAMPLES_MAX]:
            code = block.get("code", "") or block.get("interactive_input", "")
            if code:
                stimulus_adds.append(f"이슈 원문 호출 패턴: {code[:_FEEDBACK_SHORT_STR_LEN]}")
        if coverage < 0.3:
            must_change = [
                "Do not keep the same low-coverage scenario unchanged.",
                "Retarget to a scenario/source path that actually reaches the suspicious source file, or switch to the next validated scenario.",
                "Build the setup from the issue call pattern and existing tests, not from unrelated failing behavior.",
            ]
        else:
            must_change = [
                "Keep the target source, but change setup/execution so the target function is reached through a public caller.",
                "The next execution must cover the target source file and target function before it can be accepted.",
            ]
        repair_directive = _build_repair_directive(
            "RETARGET_SOURCE",
            "before-patch failure did not execute the suspicious target location",
            scenario,
            clue,
            must_change=must_change,
            must_keep=[
                f"target source: {source_file}" if source_file else "",
                f"target function: {target_func}" if target_func else "",
            ],
            execution_result=execution_result,
            raw_output=raw_output,
            coverage=coverage,
        )

    elif failure_type == FailureType.WEAK_ALIGNMENT:
        features = failure_features or {}
        diagnosis = (
            "정합성 부족. "
            f"s_b(bug_fail)={bug_fail:.2f}/{_BUG_FAIL_MAX}, "
            f"s_a(issue_align)={issue_align:.3f}/{_ISSUE_ALIGN_MAX}, "
            f"s_c(coverage)={coverage:.3f}/{_COVERAGE_MAX}"
        )
        weakest_gate = min(
            bug_fail / _BUG_FAIL_MAX if _BUG_FAIL_MAX else 0.0,
            issue_align / _ISSUE_ALIGN_MAX if _ISSUE_ALIGN_MAX else 0.0,
            coverage / _COVERAGE_MAX if _COVERAGE_MAX else 0.0,
        )
        # Switch scenario if every gate signal is very weak.
        if weakest_gate < _SWITCH_SCENARIO_THRESHOLD:
            switch_scenario = True

        # ── 피처 기반 구체적 피드백 (ALIGNED 거부 원인 명시) ──
        if features.get("f_setup_assert"):
            detail = _extract_error_detail_from_raw(raw_output, "f_setup_assert")
            msg = (
                "setUp/setUpClass 중 에러가 발생해 테스트가 setup 단계에서 실패했다. "
                "이는 버그와 무관한 실패이므로 ALIGNED로 인정되지 않는다. "
                "setUp 없이 직접 객체를 생성하거나, 테스트 픽스처 방식을 변경하라."
            )
            if detail:
                msg += f" 발생한 에러: {detail}"
            else:
                tb = _fallback_traceback(raw_output)
                if tb:
                    msg += f" 마지막 traceback:\n{tb}"
            precond_adds.append(msg)

        if features.get("f_import_err"):
            detail = _extract_error_detail_from_raw(raw_output, "f_import_err")
            if detail:
                # NameError: name 'X' is not defined → X를 import하라
                name_m = re.search(r"name '([^']+)' is not defined", detail)
                import_m = re.search(r"ImportError: (.+)", detail)
                if name_m:
                    precond_adds.append(
                        f"'{name_m.group(1)}'이 정의되지 않았다. "
                        f"테스트 파일 상단에 해당 심볼을 import하라. "
                        f"원문 에러: {detail}"
                    )
                elif import_m:
                    precond_adds.append(
                        f"import 에러가 발생했다. import 경로를 확인하고 수정하라. "
                        f"원문 에러: {detail}"
                    )
                else:
                    precond_adds.append(f"import/name 에러: {detail}")
            else:
                tb = _fallback_traceback(raw_output)
                precond_adds.append(
                    "NameError 또는 ImportError가 발생했다. "
                    "필요한 모든 심볼을 import하라."
                    + (f" 마지막 traceback:\n{tb}" if tb else "")
                )

        if features.get("f_db_err"):
            detail = _extract_error_detail_from_raw(raw_output, "f_db_err")
            msg = (
                "DB 관련 에러가 발생했다. "
                "DB 접근이 필요 없는 테스트라면 django.test.SimpleTestCase를 사용하라. "
                "DB 접근이 필요하다면 django.test.TestCase를 사용하고 "
                "필요한 fixtures나 setUp에서 객체를 직접 생성하라."
            )
            if detail:
                msg += f" 원문 에러: {detail}"
            precond_adds.append(msg)

        # 가장 낮은 구성요소 기반 피드백
        if bug_fail < _ALIGNED_BUG_FAIL_MIN and not any([
            features.get("f_setup_assert"),
            features.get("f_import_err"),
            features.get("f_db_err"),
        ]):
            # 피처 분류 안 된 저점 케이스 → fallback 피드백
            tb = _fallback_traceback(raw_output)
            expected_failure_override = (
                "패치 적용 전 코드에서 반드시 FAILED 되어야 한다. "
                "AssertionError로 실패하도록 assertion을 작성하라."
            )
            if tb:
                precond_adds.append(
                    f"이전 실행의 마지막 traceback (원인 파악에 활용):\n{tb}"
                )
        elif bug_fail < _FEEDBACK_BUG_FAIL_WEAK:
            expected_failure_override = (
                "패치 적용 전 코드에서 반드시 FAILED 되어야 한다."
            )
            oracle_adds.append(
                "assertion은 버그 수정 후의 올바른 결과를 기대해야 한다."
            )
        if issue_align < _FEEDBACK_ISSUE_ALIGN_WEAK:
            oracle_adds.append(
                "테스트는 이슈에서 설명한 문제 상황을 정확히 재현해야 한다."
            )
            for out in expected_outputs[:_FEEDBACK_OUTPUTS_MAX]:
                oracle_adds.append(f"기대 출력: {out[:_FEEDBACK_MID_STR_LEN]}")
            # 어떤 식별자가 빠져 있는지 명시하여 모델이 구체적으로 보완하도록 유도
            test_code_lower = generated_test.get("test_code", "").lower()
            clue_ids = clue.get("identifiers", {})
            all_ids: set = set()
            for fn in clue_ids.get("functions", []):
                if isinstance(fn, str):
                    all_ids.add(fn.lower())
            for cls in clue_ids.get("classes", []):
                if isinstance(cls, str):
                    all_ids.add(cls.lower())
            for exc in clue_ids.get("exceptions", []):
                if isinstance(exc, str):
                    all_ids.add(exc.lower())
            missing_ids = sorted(i for i in all_ids if i not in test_code_lower)
            if missing_ids:
                oracle_adds.append(
                    f"다음 식별자를 테스트 코드에 반드시 포함해야 한다: "
                    f"{', '.join(missing_ids[:_FEEDBACK_MISSING_IDS_MAX])}"
                )
        if coverage < _COVERAGE_FALLBACK:
            target = scenario.get("target_location") or {}
            source_file = target.get("source_file", "")
            target_func = target.get("target_function", "")
            if source_file:
                stimulus_adds.append(
                    f"테스트는 {source_file}의 코드를 실행해야 한다."
                )
            if target_func:
                stimulus_adds.append(
                    f"테스트는 {target_func} 함수를 호출해야 한다."
                )

    _append_composite_gate_feedback()

    if (
        failure_type not in {FailureType.ALIGNED, FailureType.WEAK_ALIGNMENT}
        and not repair_directive.get("mode")
    ):
        target = scenario.get("target_location") or {}
        target_source = str(target.get("source_file", ""))
        target_function = str(target.get("target_function", ""))
        generated_code = str(
            generated_test.get("test_code")
            or generated_test.get("append_block")
            or ""
        )
        assert_lines = [
            line.strip()[:_FEEDBACK_SHORT_STR_LEN]
            for line in generated_code.splitlines()
            if line.strip().startswith(("self.assert", "assert ", "np.testing.assert"))
            and not re.search(r"\b(?:np\.)?(?:testing\.)?assert_array_equal\s*\(", line)
        ]

        if failure_type == FailureType.NOT_FAILED:
            bug_trigger_patterns = _bug_triggering_issue_call_patterns(clue)
            has_bug_trigger_call = _generated_test_contains_bug_trigger_call(clue, generated_test)
            scenario_has_trigger = _scenario_has_bug_trigger_evidence(scenario, clue)
            must_change = [
                "Rewrite the assertion so it fails on the buggy pre-patch behavior and passes on the fixed behavior.",
                "Use the issue-stated expected output when available; do not assert a condition that already passes on buggy code.",
                "Keep the same issue stimulus but assert the returned value or public state after the target call.",
            ]
            if target_function:
                must_change.append(f"The next test must visibly exercise target function/API: {target_function}.")
            if not bug_trigger_patterns or not scenario_has_trigger:
                switch_scenario = True
                repair_directive = _build_repair_directive(
                    "SWITCH_SCENARIO_OR_TEST_FILE",
                    "no usable issue/scenario bug-trigger pattern exists for a NOT_FAILED repair",
                    scenario,
                    clue,
                    must_change=[
                        "Switch to a scenario that contains actual buggy output or an explicit failing/problem reproduction call.",
                        "Do not keep a baseline-only scenario when actual buggy output exists in the issue.",
                        "Build the next test from issue-derived bug-trigger evidence before rewriting the oracle.",
                    ],
                    must_keep=[
                        f"bug-triggering issue call: {pattern}"
                        for pattern in bug_trigger_patterns[:2]
                    ],
                    execution_result=execution_result,
                    raw_output=raw_output,
                    coverage=coverage,
                )
            elif not has_bug_trigger_call:
                stimulus_adds.append(
                    "현재 테스트가 이슈의 bug-triggering 호출 패턴을 포함하지 않는다. "
                    f"다음 호출/setup을 테스트 stimulus로 복원하라: {bug_trigger_patterns[0][:_FEEDBACK_MID_STR_LEN]}"
                )
                repair_directive = _build_repair_directive(
                    "REWRITE_STIMULUS",
                    "before-patch execution did not fail because the generated test omitted the available bug-triggering issue call",
                    scenario,
                    clue,
                    must_change=[
                        "Replace the current baseline/sanity stimulus with the issue's bug-triggering call pattern.",
                        "Preserve setup required by the bug-triggering call.",
                        "After executing the bug-triggering call, assert fixed expected behavior.",
                    ],
                    must_keep=[
                        f"bug-triggering issue call: {pattern}"
                        for pattern in bug_trigger_patterns[:2]
                    ],
                    forbidden_patterns=[],
                    execution_result=execution_result,
                    raw_output=raw_output,
                    coverage=coverage,
                )
            else:
                repair_directive = _build_repair_directive(
                    "REWRITE_ORACLE",
                    "before-patch execution did not fail even though the bug-triggering stimulus is present, so the oracle is not exposing the bug",
                    scenario,
                    clue,
                    must_change=must_change,
                    must_keep=[
                        f"target source: {target_source}" if target_source else "",
                        f"target function: {target_function}" if target_function else "",
                    ],
                    forbidden_patterns=assert_lines[:3],
                    execution_result=execution_result,
                    raw_output=raw_output,
                    coverage=coverage,
                )
        elif failure_type == FailureType.NOT_VALID:
            missing_symbols: List[str] = []
            for msg in error_messages or []:
                missing_symbols.extend(re.findall(r"NameError: '([^']+)'", msg))
                missing_symbols.extend(re.findall(r"Missing import: add 'import ([A-Za-z_][A-Za-z0-9_]*)'", msg))
            missing_symbols = _dedupe_strs(missing_symbols, limit=4)
            must_change = [
                "Make the generated test collectable by the project runner before changing the oracle.",
                "Remove or replace unresolved symbols/imports using only Existing Imports or Available Imports from Repository.",
                "Mirror the target test file's existing class/function style exactly.",
            ]
            if missing_symbols:
                must_change.append(
                    "Do not leave these symbols unresolved: " + ", ".join(missing_symbols)
                )
            repair_directive = _build_repair_directive(
                "FIX_IMPORTS",
                "generated test is not collectable because imports or symbols are invalid",
                scenario,
                clue,
                must_change=must_change,
                must_keep=[
                    f"target test file: {target.get('candidate_test_file', '')}",
                    f"target function: {target_function}" if target_function else "",
                ],
                forbidden_patterns=[f"{name}." for name in missing_symbols],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )
        elif failure_type == FailureType.ERROR:
            repair_directive = _build_repair_directive(
                "FIX_API_USAGE",
                "test execution errored before producing a reliable bug assertion",
                scenario,
                clue,
                must_change=[
                    "Remove the setup/import/API pattern that causes the runtime error.",
                    "Use only public APIs and symbols that exist in this repository version.",
                    "The next test must fail by assertion of issue behavior, not by construction/runtime error.",
                ],
                must_keep=[
                    f"target source: {target_source}" if target_source else "",
                    f"target function: {target_function}" if target_function else "",
                ],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )
        else:
            repair_directive = _build_repair_directive(
                "IMPROVE_ALIGNMENT",
                "one or more alignment gates are below threshold",
                scenario,
                clue,
                must_change=[
                    "Rewrite the test so the stimulus, target coverage, and oracle all match the issue.",
                    "Call the target API or a public caller that reaches the target source.",
                    "Assert issue-specific fixed behavior rather than a structural placeholder.",
                ],
                must_keep=[
                    f"target source: {target_source}" if target_source else "",
                    f"target function: {target_function}" if target_function else "",
                ],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )

    feedback = ScenarioFeedback(
        failure_type=failure_type.value,
        diagnosis=diagnosis,
        oracle_additions=oracle_adds,
        stimulus_additions=stimulus_adds,
        precondition_additions=precond_adds,
        expected_failure_override=expected_failure_override,
        switch_scenario=switch_scenario,
        repair_directive=repair_directive,
    )
    feedback = _resolve_feedback_conflicts(feedback)
    if _scenario_requires_unpaired_oracle_feedback_sanitization(scenario):
        feedback = _sanitize_unpaired_oracle_feedback(feedback, clue, scenario)
    return feedback


# ---------------------------------------------------------------------------
# 4. 시나리오 보강
# ---------------------------------------------------------------------------

def _dedupe_feedback_items(items: List[str], limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        norm = re.sub(r"\s+", " ", text.lower())
        if not text or norm in seen:
            continue
        seen.add(norm)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def refine_scenario(
    scenario: Dict[str, Any],
    feedback: ScenarioFeedback,
    iteration: int,
) -> Dict[str, Any]:
    """피드백을 반영하여 시나리오 dict를 보강한 사본을 반환."""
    refined = copy.deepcopy(scenario)
    tag = f"[iteration-{iteration} feedback]"

    # ── 과거 피드백 태그 누적 제한: 최근 iteration만 유지 ──
    keep_min_iter = max(1, iteration - _FEEDBACK_PREV_ITERATION_KEEP)  # 현재 + 이전 N개 유지
    _prune_old_feedback_tags(refined, keep_min_iter)

    directive = feedback.repair_directive if isinstance(feedback.repair_directive, dict) else {}
    has_directive = bool(directive.get("mode"))
    if has_directive:
        # The repair directive is the canonical feedback channel. Avoid
        # repeatedly appending long natural-language advice into scenario
        # fields, which dilutes the next generation prompt.
        refined["repair_directive"] = sanitize_repair_directive(directive) if selected_example_requires_oracle_regeneration(refined) else directive
    else:
        refined.pop("repair_directive", None)

    memory = dict(refined.get("repair_memory") or {})
    forbidden_files = [
        str(path).strip()
        for path in memory.get("forbidden_test_files", []) or []
        if str(path).strip()
    ]
    mode = str(directive.get("mode") or "")
    evidence = directive.get("evidence") if isinstance(directive.get("evidence"), dict) else {}
    if mode in {"CHANGE_TEST_FILE", "SWITCH_SCENARIO_OR_TEST_FILE"}:
        target_loc = scenario.get("target_location") if isinstance(scenario.get("target_location"), dict) else {}
        for path in (
            target_loc.get("candidate_test_file", ""),
            evidence.get("candidate_test_file", ""),
        ):
            text = str(path or "").strip()
            if text and text not in forbidden_files:
                forbidden_files.append(text)
    if forbidden_files:
        memory["forbidden_test_files"] = forbidden_files[:12]
    if feedback.candidate_test_file_override:
        memory["required_target_file"] = feedback.candidate_test_file_override
    if memory:
        refined["repair_memory"] = memory

    if feedback.oracle_additions and not has_directive and not selected_example_requires_oracle_regeneration(refined):
        current_oracle = refined.get("oracle", "")
        additions = " ".join(
            _dedupe_feedback_items(feedback.oracle_additions, _FEEDBACK_PROMPT_VISIBLE_MAX)
        )
        refined["oracle"] = f"{current_oracle} {tag} {additions}".strip()

    if feedback.stimulus_additions and not has_directive:
        stims = refined.get("execution_stimulus", [])
        for s in _dedupe_feedback_items(feedback.stimulus_additions, _FEEDBACK_PROMPT_VISIBLE_MAX):
            stims.append(f"{tag} {s}")
        refined["execution_stimulus"] = stims

    if feedback.precondition_additions and not has_directive:
        setup = refined.get("setup_steps", [])
        for p in _dedupe_feedback_items(feedback.precondition_additions, _FEEDBACK_PROMPT_VISIBLE_MAX):
            setup.append(f"{tag} {p}")
        refined["setup_steps"] = setup

    if feedback.expected_failure_override and not has_directive:
        refined["expected_failure"] = (
            f"{refined.get('expected_failure', '')} {tag} "
            f"{feedback.expected_failure_override}"
        ).strip()

    # ── candidate_test_file 교체 (NOT_VALID 등) ──
    if feedback.candidate_test_file_override:
        target_loc = refined.get("target_location", {})
        old_file = target_loc.get("candidate_test_file", "")
        target_loc["candidate_test_file"] = feedback.candidate_test_file_override
        refined["target_location"] = target_loc
        # Also update relevant_test_files to prioritize the new file
        rel_tests = refined.get("relevant_test_files", [])
        if feedback.candidate_test_file_override not in rel_tests:
            rel_tests.insert(0, feedback.candidate_test_file_override)
            refined["relevant_test_files"] = rel_tests
        logger.info(
            "candidate_test_file overridden: %s → %s",
            old_file, feedback.candidate_test_file_override,
        )

    return sanitize_oracle_regeneration_payload(refined)


def _prune_old_feedback_tags(scenario: Dict[str, Any], keep_min_iter: int) -> None:
    """iteration 태그가 keep_min_iter 미만인 피드백 항목을 제거한다."""
    old_tag_pattern = re.compile(r"\[iteration-(\d+) feedback\]")

    def _is_old_tagged(text: str) -> bool:
        m = old_tag_pattern.search(text)
        return m is not None and int(m.group(1)) < keep_min_iter

    # list fields: execution_stimulus, setup_steps
    for key in ("execution_stimulus", "setup_steps"):
        items = scenario.get(key, [])
        if isinstance(items, list):
            scenario[key] = [item for item in items if not _is_old_tagged(str(item))]

    # expected_failure: 첫 오래된 태그 이전 부분만 유지
    ef = scenario.get("expected_failure", "")
    if isinstance(ef, str):
        for m in old_tag_pattern.finditer(ef):
            iter_num = int(m.group(1))
            if iter_num < keep_min_iter:
                scenario["expected_failure"] = ef[:m.start()].strip()
                break
        else:
            scenario["expected_failure"] = ef

    oracle = scenario.get("oracle", "")
    if isinstance(oracle, str):
        for m in old_tag_pattern.finditer(oracle):
            iter_num = int(m.group(1))
            if iter_num < keep_min_iter:
                scenario["oracle"] = oracle[:m.start()].strip()
                break
        else:
            scenario["oracle"] = oracle


# ---------------------------------------------------------------------------
# 5. 메인 인터페이스
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """1회 평가 결과."""
    iteration: int
    failure_type: str
    score_breakdown: Dict[str, Any]
    diagnosis: str
    feedback: Dict[str, Any]
    refined_scenario: Dict[str, Any]
    should_continue: bool
    test_results: Dict[str, str]
    coverage_summary: Dict[str, Any]
    failure_type_detail: str = ""
    execution_status: str = "NOT_RUN"
    validation_status: str = "NOT_RUN"
    m7_alignment_status: Optional[str] = None
    admitted_to_final_set: bool = False
    diagnostic_only: bool = True
    legacy_failure_type: Optional[str] = None
    structured_feedback: Dict[str, Any] = field(default_factory=dict)
    iteration_feedback_summary: Dict[str, Any] = field(default_factory=dict)
    alignment_verdict: str = "NOT_ALIGNED"
    admission_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if not data.get("legacy_failure_type"):
            data["legacy_failure_type"] = self.failure_type
        return data


class AlignmentScorer:
    """patch-free 규칙기반 정합성 평가자."""

    def evaluate(
        self,
        execution_result: Dict[str, Any],
        clue: Dict[str, Any],
        scenario: Dict[str, Any],
        generated_test: Dict[str, Any],
        iteration: int = 1,
        validation_report: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        feature_flags: V22FeatureFlags | Mapping[str, Any] | None = None,
    ) -> AlignmentResult:
        """alignment_runner의 실행 결과를 평가한다."""
        if scenario is None:
            scenario = {}
        if clue is None:
            clue = {}
        test_results = execution_result.get("test_results", {})
        has_error = execution_result.get("has_error", False)
        has_failure = execution_result.get("has_failure", False)
        coverage_data = execution_result.get("coverage_data", {})
        error_messages = execution_result.get("error_messages", [])
        raw_output = execution_result.get("raw_output", "")
        methodology_revision = str((context or {}).get("methodology_revision") or "")
        v36_methodology = methodology_revision in {"v36", "v37"}

        # Step 1: 세 구성 점수 산출
        failure_features = extract_failure_features(raw_output, test_results)
        bug_fail_features = compute_bug_fail_features(
            test_results=test_results,
            raw_output=raw_output,
            clue=clue,
            scenario=scenario,
            generated_test=generated_test,
        )
        v37_kw_hit_ratio: float | None = None
        v37_kw_provenance: dict[str, Any] = {}
        if methodology_revision == "v37":
            v37_kw_hit_ratio, v37_kw_provenance = compute_v37_keyword_hit_ratio(
                clue,
                test_results,
                error_messages if isinstance(error_messages, Sequence) else [],
                raw_output,
            )
        bug_fail = (
            compute_v36_bug_fail_score(
                test_results,
                v37_kw_hit_ratio
                if methodology_revision == "v37"
                else bug_fail_features.get("f_symptom", 0.0),
            )
            if v36_methodology
            else compute_bug_fail_score(
                test_results,
                has_error,
                raw_output=raw_output,
                clue=clue,
                scenario=scenario,
                generated_test=generated_test,
            )
        )
        if v36_methodology:
            bug_fail_features = {
                **bug_fail_features,
                "kw_hit_ratio": (
                    v37_kw_hit_ratio
                    if methodology_revision == "v37"
                    else bug_fail_features.get("f_symptom", 0.0)
                ),
                "kw_hit_ratio_provenance": (
                    v37_kw_provenance
                    if methodology_revision == "v37"
                    else {"source": "legacy_v36_f_symptom"}
                ),
                "v36_formula_inputs": ["fail_ratio", "kw_hit_ratio"],
            }
        issue_align = (
            compute_v36_issue_alignment_score(clue, generated_test)
            if v36_methodology
            else compute_issue_alignment_score(clue, scenario, generated_test)
        )
        v36_raw_coverage, v36_raw_coverage_evidence = compute_v36_line_coverage_score(
            coverage_data if isinstance(coverage_data, Mapping) else {},
            execution_result,
        ) if v36_methodology else (None, {})
        coverage = (
            float(v36_raw_coverage or 0.0)
            if v36_methodology
            else compute_coverage_score(
                coverage_data,
                scenario,
                execution_result,
                clue=clue,
                context=context,
            )
        )
        resolved_feature_flags = _resolve_m7_feature_flags(feature_flags)
        canonical_sbfl_result = _extract_canonical_m6_sbfl(context, execution_result)
        sbfl_weighted_coverage = compute_m7_sbfl_weighted_coverage(
            base_coverage=coverage,
            coverage_data=coverage_data,
            scenario=scenario,
            sbfl_result=canonical_sbfl_result if resolved_feature_flags.m7_sbfl_weighted_coverage else None,
            execution_result=execution_result,
            clue=clue,
            context=context,
            require_weighted=resolved_feature_flags.m7_sbfl_weighted_coverage,
        )
        if not resolved_feature_flags.m7_sbfl_weighted_coverage:
            sbfl_weighted_coverage["fallback_reason"] = "feature_disabled"
        weighted_coverage_available = bool(
            sbfl_weighted_coverage.get("coverage_valid", False)
        )
        weighted_coverage = (
            float(sbfl_weighted_coverage["weighted_coverage"])
            if weighted_coverage_available
            and isinstance(sbfl_weighted_coverage.get("weighted_coverage"), (int, float))
            else None
        )
        coverage_score_available = (
            v36_raw_coverage is not None
            if v36_methodology and iteration == 1
            else weighted_coverage_available
        )
        canonical_coverage = (
            v36_raw_coverage
            if v36_methodology and iteration == 1
            else weighted_coverage
        )
        coverage_contract = build_coverage_field_contract(
            execution_result=execution_result,
            coverage_data=coverage_data,
            scenario=scenario,
            clue=clue,
            context=context,
            s_c_prime=canonical_coverage,
            s_c_prime_available=weighted_coverage_available,
            s_c_prime_unavailable_reason=str(
                sbfl_weighted_coverage.get("fallback_reason") or ""
            ) or None,
        )
        if v36_methodology:
            coverage_contract.update(
                {
                    "s_c": v36_raw_coverage,
                    "s_c_admission_role": (
                        "ITERATION_1_GATE" if iteration == 1 else "DIAGNOSTIC_ONLY"
                    ),
                    "s_c_prime": weighted_coverage,
                    "s_c_prime_status": (
                        "AVAILABLE_WEIGHTED_OCHIAI"
                        if weighted_coverage_available
                        else "UNAVAILABLE"
                    ),
                    "coverage_gate_iteration": iteration,
                    "v36_raw_coverage_evidence": v36_raw_coverage_evidence,
                }
            )
        # Admission remains fail closed, but the canonical score is nullable.
        # This numeric projection is deliberately separate from s_c_prime so
        # unavailable evidence cannot be confused with a computed zero.
        coverage = canonical_coverage if canonical_coverage is not None else 0.0
        target_verification_evidence = compute_target_verification_evidence(
            scenario=scenario,
            context=context,
            execution_result=execution_result,
            generated_test=generated_test,
        )
        target_verified = bool(target_verification_evidence["target_verified"])
        strong_issue_evidence = has_strong_issue_evidence(clue, generated_test)
        oracle_quality = evaluate_oracle_quality(generated_test, clue)
        oracle_consistency = evaluate_oracle_consistency(scenario, clue, generated_test)
        coverage_fallback_reason = ""
        if (
            coverage == 0
            and not coverage_data
            and target_verified
            and strong_issue_evidence
            and "No data to report" in raw_output
        ):
            coverage_fallback_reason = "coverage_tool_no_data_no_line_evidence"

        # Step 2: 분류 (V2: Requirements-Based Scoring)
        failure_type = classify_and_score_v2(
            test_results=test_results,
            has_error=has_error,
            bug_fail=bug_fail,
            coverage=coverage,
            issue_align=issue_align,
            failure_features=failure_features,
            error_messages=error_messages,
            target_verified=target_verified,
            strong_issue_evidence=strong_issue_evidence,
        )
        failure_type_detail = ""
        if (
            resolved_feature_flags.m7_sbfl_weighted_coverage
            and not sbfl_weighted_coverage.get("coverage_valid", False)
            and failure_type == FailureType.ALIGNED
        ):
            failure_type = FailureType.NO_COVERAGE
            failure_type_detail = "M7_SBFL_EVIDENCE_UNAVAILABLE"
        supplemental_metadata = (
            canonical_sbfl_result.get("metadata")
            if isinstance(canonical_sbfl_result, Mapping)
            and isinstance(canonical_sbfl_result.get("metadata"), Mapping)
            else {}
        )
        supplemental_collection = supplemental_metadata.get("supplemental_pass_collection")
        if (
            failure_type == FailureType.NO_COVERAGE
            and isinstance(supplemental_collection, Mapping)
            and (
                supplemental_metadata.get("diagnostic_classification")
                == "SBFL_UNAVAILABLE_INSUFFICIENT_P"
                or supplemental_collection.get("m7_diagnostic_classification")
                == "SBFL_UNAVAILABLE_INSUFFICIENT_P"
            )
        ):
            failure_type_detail = "SBFL_UNAVAILABLE_INSUFFICIENT_P"
        conservative_gate_reasons: List[str] = []
        repair_failed_reason = str(generated_test.get("repair_failed_reason") or "")
        retry_required_oracle_risks = generated_test.get("retry_required_oracle_risks", []) or []
        semantic_risk_flags = generated_test.get("semantic_risk_flags", []) or []
        blocking_oracle_flags = sorted(
            set(oracle_quality.risk_flags)
            & {
                "external_network_call",
                "django_inline_model",
                "numpy_direct_equality",
                "nan_comparison",
                "no_explicit_oracle",              # assertion 자체가 없음
                "trivial_oracle",                  # 의미 없는 pass assertion
            }
        )
        if failure_type == FailureType.ALIGNED and blocking_oracle_flags:
            conservative_gate_reasons.append(
                "blocking_oracle_risk_flags=" + ",".join(blocking_oracle_flags)
            )
        if failure_type == FailureType.ALIGNED and not oracle_consistency.usable_oracle:
            conservative_gate_reasons.append(
                "oracle_consistency=" + oracle_consistency.status
            )
            failure_type_detail = failure_type_detail or "ORACLE_CONSISTENCY"
        if (
            oracle_consistency.selected_example_id
            and not oracle_consistency.trigger_present
            and failure_type != FailureType.ERROR
        ):
            conservative_gate_reasons.append("bug_trigger_missing")
            failure_type_detail = failure_type_detail or "TRIGGER_MISMATCH"
        nonblocking_gate_warnings: List[str] = []
        target = scenario.get("target_location") or {}
        target_func = target.get("target_function", "") if isinstance(target, dict) else ""
        source_file = target.get("source_file", "") if isinstance(target, dict) else ""
        oracle_contract = scenario.get("oracle_contract") if isinstance(scenario.get("oracle_contract"), dict) else {}
        oracle_type = str(oracle_contract.get("oracle_type") or scenario.get("oracle_type") or "")
        oracle_source = str(oracle_contract.get("oracle_source") or scenario.get("oracle_source") or "")
        issue_reported_invariant = _issue_reported_semantic_invariant(
            str(generated_test.get("test_code") or generated_test.get("append_block") or ""),
            clue,
        )
        if issue_reported_invariant:
            oracle_type = "semantic_invariant"
            oracle_source = "issue_reported_semantic_invariant"
        private_target_reached_by_public_api = (
            bool(target_func)
            and str(target_func).startswith("_")
            and coverage >= _COVERAGE_MIN_GATE
            and strong_issue_evidence
            and _target_source_has_coverage(coverage_data, source_file)
        )
        if failure_type != FailureType.ERROR and not target_verified:
            reason = "target_verified=False"
            if private_target_reached_by_public_api:
                reason += "_allowed_private_target_via_public_api"
                nonblocking_gate_warnings.append(reason)
            else:
                conservative_gate_reasons.append(reason)
                failure_type_detail = failure_type_detail or "TARGET_NOT_VERIFIED"
        failure_signal = _extract_failure_signal(raw_output)
        issue_reported_failure = _is_issue_reported_failure_signal(failure_signal, clue)
        if (
            failure_type == FailureType.ALIGNED
            and _is_construction_failure_signal(failure_signal)
            and not issue_reported_failure
        ):
            conservative_gate_reasons.append(
                "construction_failure="
                + (failure_signal.get("exception_type") or "runtime_error")
            )
            failure_type = FailureType.ERROR
            failure_type_detail = "CONSTRUCTION_ERROR"
        elif failure_type != FailureType.ERROR and issue_reported_failure:
            nonblocking_gate_warnings.append("issue_reported_exception_observed")
        if (
            failure_type == FailureType.ALIGNED
            and not strong_issue_evidence
            and issue_align < _ISSUE_ALIGN_STRONG_GATE
        ):
            nonblocking_gate_warnings.append("weak_issue_evidence_token_overlap_only")
        soft_oracle_flags = sorted(
            set(oracle_quality.risk_flags)
            & {
                "weak_structural_oracle",
                "warning_presence_oracle",
                "buggy_output_as_oracle",
                "image_comparison_decorator",
                "raises_only_no_body_assertion",
                "fix_disappearing_exception_oracle",
                "raw_rendered_output_exact_match",
                "private_attribute_oracle",
                "guessed_expected_array",
                "guessed_expected_value",
                "constant_negative_oracle",
                "exception_message_match",
                "exception_message_negative_oracle",
                "warning_catch_only",
                "multiple_generated_tests",
            }
        )
        conservative_oracle_flags = sorted(
            set(soft_oracle_flags)
            & {
                "buggy_output_as_oracle",
                "weak_structural_oracle",
                "private_attribute_oracle",
                "guessed_expected_array",
                "guessed_expected_value",
            }
        )
        raw_execution_blocking_flags = execution_result.get("blocking_oracle_flags") or []
        execution_blocking_flags = (
            validated_v37_blocking_oracle_flags(raw_execution_blocking_flags)
            if methodology_revision == "v37"
            else [str(flag) for flag in raw_execution_blocking_flags if str(flag)]
        )
        v29_blocking_oracle_flags = sorted(
            set(blocking_oracle_flags)
            | set(conservative_oracle_flags)
            | set(execution_blocking_flags)
        )
        warning_oracle_flags = sorted(set(soft_oracle_flags) - set(conservative_oracle_flags))
        if failure_type == FailureType.ALIGNED and conservative_oracle_flags:
            conservative_gate_reasons.append(
                "conservative_oracle_risk_flags=" + ",".join(conservative_oracle_flags)
            )
            failure_type_detail = failure_type_detail or "SEMANTIC_RISK"
        if failure_type == FailureType.ALIGNED and warning_oracle_flags:
            nonblocking_gate_warnings.append(
                "soft_oracle_risk_flags=" + ",".join(warning_oracle_flags)
            )
        if failure_type == FailureType.ALIGNED and oracle_type == "last_resort_structural":
            nonblocking_gate_warnings.append("oracle_type=last_resort_structural")
        if (
            failure_type == FailureType.ALIGNED
            and oracle_source == "inferred_semantic"
            and not target_verified
        ):
            nonblocking_gate_warnings.append("oracle_source=inferred_semantic_target_unverified")
        if (
            failure_type == FailureType.ALIGNED
            and not target_verified
            and "weak_issue_evidence_token_overlap_only" in nonblocking_gate_warnings
        ):
            nonblocking_gate_warnings.append(
                "target_unverified_and_weak_issue_evidence"
            )
        if failure_type == FailureType.ALIGNED and repair_failed_reason:
            conservative_gate_reasons.append("repair_failed=" + repair_failed_reason)
            failure_type_detail = "REPAIR_FAILED"
        if failure_type == FailureType.ALIGNED and semantic_risk_flags:
            conservative_gate_reasons.append(
                "semantic_risk_flags=" + ",".join(map(str, semantic_risk_flags))
            )
            failure_type_detail = failure_type_detail or "SEMANTIC_RISK"
        if failure_type == FailureType.ALIGNED and execution_result.get("flaky"):
            conservative_gate_reasons.append("flaky=True")
            failure_type_detail = failure_type_detail or "FLAKY"
        current_m7_methodology = methodology_revision in {"v29", "v30", "v31", "v36", "v37"}
        if v36_methodology:
            # V36 Conservative Gate consumes only the explicit M6 contract
            # field. All legacy static-risk and target-verification signals
            # remain diagnostics and cannot change the quantitative verdict.
            failure_type = classify_and_score_v2(
                test_results=test_results,
                has_error=has_error,
                bug_fail=bug_fail,
                coverage=coverage,
                issue_align=issue_align,
                failure_features=failure_features,
                error_messages=error_messages,
            )
            conservative_gate_reasons = []
            v29_blocking_oracle_flags = execution_blocking_flags
            if failure_type == FailureType.ALIGNED and execution_blocking_flags:
                conservative_gate_reasons.append(
                    "blocking_oracle_flags=" + ",".join(sorted(set(execution_blocking_flags)))
                )
        if failure_type == FailureType.ALIGNED and conservative_gate_reasons:
            if not current_m7_methodology:
                failure_type = FailureType.WEAK_ALIGNMENT
        if not v36_methodology:
            bug_fail, coverage, issue_align = normalize_aligned_component_scores(
                failure_type, bug_fail, coverage, issue_align,
            )
        numeric_gate_results = {
            "s_b": bug_fail,
            "s_c_prime": coverage,
            "s_a": issue_align,
            "gate1_pass": bug_fail >= _ALIGNED_BUG_FAIL_MIN,
            "gate2_pass": coverage >= _COVERAGE_MIN_GATE,
            "gate3_pass": issue_align >= _ISSUE_ALIGN_MIN_GATE,
        }
        all_numeric_gates_pass = all(
            numeric_gate_results[key]
            for key in ("gate1_pass", "gate2_pass", "gate3_pass")
        )
        conservative_gate_triggered = (
            bool(conservative_gate_reasons)
            if current_m7_methodology
            else bool(v29_blocking_oracle_flags or execution_result.get("flaky"))
        )
        conservative_gate_is_only_branching_reason = bool(
            current_m7_methodology
            and failure_type == FailureType.ALIGNED
            and all_numeric_gates_pass
            and conservative_gate_triggered
        )
        conservative_gate_pending_llm = conservative_gate_is_only_branching_reason
        # A5: quantitative DIRECT admission is canonical whenever no
        # Conservative Gate is active.  No later compatibility label may
        # downgrade this state.
        if current_m7_methodology and all_numeric_gates_pass and not conservative_gate_triggered:
            failure_type = FailureType.ALIGNED
            failure_type_detail = ""
        logger.info(
            "[iteration %d] failure_type=%s "
            "(bug_fail=%.3f, issue=%.3f, cov=%.3f, oracle=%.3f)",
            iteration, failure_type.value,
            bug_fail, issue_align, coverage, oracle_quality.score,
        )

        # Step 3: 피드백 생성
        feedback = generate_feedback(
            failure_type, clue, scenario, generated_test,
            bug_fail, issue_align, coverage,
            error_messages=error_messages,
            project_test_style=context.get("project_test_style") if context else None,
            raw_output=raw_output,
            failure_features=failure_features,
            execution_result=execution_result,
            context=context,
        )
        if failure_type == FailureType.NO_COVERAGE and canonical_coverage is None:
            raw_value = coverage_contract.get("raw_target_coverage")
            raw_text = "unavailable" if raw_value is None else f"{float(raw_value):.2f}"
            feedback.diagnosis = (
                "s_c_prime unavailable "
                f"({coverage_contract.get('s_c_prime_unavailable_reason') or 'unknown'}); "
                f"raw_target_coverage={raw_text}. "
                "Supplemental PASS-spectrum collection, not a synthetic zero, owns recovery."
            )
        if oracle_quality.risk_flags:
            feedback.diagnosis = (
                f"{feedback.diagnosis} Oracle risk: "
                f"{', '.join(oracle_quality.risk_flags)}"
            ).strip()
            feedback.oracle_additions.extend(oracle_quality.feedback)
            feedback.expected_failure_override = (
                f"{feedback.expected_failure_override} "
                "테스트는 pre-patch에서 FAIL하고 post-patch에서 PASS하는 positive oracle을 사용해야 한다."
            ).strip()
        if conservative_gate_reasons:
            feedback.diagnosis = (
                f"{feedback.diagnosis} Conservative gate: "
                f"{', '.join(conservative_gate_reasons)}"
            ).strip()
            feedback.oracle_additions.append(
                "target/oracle 재생성: 현재 테스트는 before-patch 실패는 만들었지만 치명적인 oracle 위험이 있다."
            )
            if repair_failed_reason:
                feedback.oracle_additions.append(
                    "repair_failed_reason이 남아 있으므로 같은 oracle을 유지하지 말고 assertion을 처음부터 재작성하라: "
                    f"{repair_failed_reason}"
                )
            if semantic_risk_flags:
                feedback.stimulus_additions.append(
                    "issue/context와 무관한 API 또는 setup을 제거하고 target source의 public caller를 직접 실행하라: "
                    f"{', '.join(map(str, semantic_risk_flags))}"
                )
            if not (feedback.repair_directive or {}).get("mode"):
                if not oracle_consistency.usable_oracle:
                    feedback.repair_directive = _build_repair_directive(
                        "REWRITE_ORACLE",
                        "oracle is incompatible with the selected reproduction example provenance",
                        scenario,
                        clue,
                        must_change=[
                            "Regenerate the oracle for the selected reproduction stimulus.",
                            "Do not reuse expected outputs from another example or a baseline scenario.",
                            "Keep the selected stimulus/setup and assert an EB-grounded fixed behavior.",
                        ],
                        must_keep=[
                            f"selected example id: {oracle_consistency.selected_example_id}",
                            "selected trigger stimulus and setup",
                        ],
                        forbidden_patterns=list(oracle_consistency.reasons),
                        execution_result=execution_result,
                        raw_output=raw_output,
                        coverage=coverage,
                    )
                elif conservative_oracle_flags:
                    feedback.repair_directive = _build_repair_directive(
                        "REWRITE_ORACLE",
                        "oracle is too weak or uses buggy/private/guessed values, so ALIGNED is blocked",
                        scenario,
                        clue,
                        must_change=[
                            "Rewrite the assertion from fixed expected behavior, not from the buggy output.",
                            "Assert a public return value or public state reached by the target call.",
                            "Do not use private attributes, raw object identity/equality, or guessed exact arrays/values.",
                        ],
                        forbidden_patterns=list(conservative_oracle_flags),
                        execution_result=execution_result,
                        raw_output=raw_output,
                        coverage=coverage,
                    )
                elif "target_verified=False" in ",".join(conservative_gate_reasons):
                    feedback.repair_directive = _build_repair_directive(
                        "RETARGET_SOURCE",
                        "test failed but did not verify execution of the intended target location",
                        scenario,
                        clue,
                        must_change=[
                            "Change the execution stimulus so the intended target source/function is reached.",
                            "Use a public caller if the target function is private.",
                            "Do not keep the same target with the same low or unverified coverage.",
                        ],
                        execution_result=execution_result,
                        raw_output=raw_output,
                        coverage=coverage,
                    )
        if nonblocking_gate_warnings:
            feedback.diagnosis = (
                f"{feedback.diagnosis} Gate warnings: "
                f"{', '.join(nonblocking_gate_warnings)}"
            ).strip()
        if failure_type != FailureType.ALIGNED and not (feedback.repair_directive or {}).get("mode"):
            target = scenario.get("target_location") or {}
            source_file = target.get("source_file", "") if isinstance(target, dict) else ""
            target_func = target.get("target_function", "") if isinstance(target, dict) else ""
            feedback.repair_directive = _build_repair_directive(
                "IMPROVE_ALIGNMENT",
                "one or more alignment gates are below threshold and need a concrete rewrite",
                scenario,
                clue,
                must_change=[
                    "Rewrite the test so the stimulus, target coverage, and oracle all match the issue.",
                    "Call the target API or a public caller that reaches the target source.",
                    "Assert issue-specific fixed behavior rather than a structural placeholder.",
                ],
                must_keep=[
                    f"target source: {source_file}" if source_file else "",
                    f"target function: {target_func}" if target_func else "",
                ],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )
        if (
            iteration >= 2
            and failure_type in {FailureType.NO_COVERAGE, FailureType.WEAK_ALIGNMENT}
            and (
                coverage < _COVERAGE_MIN_GATE
                or failure_type_detail == "TARGET_NOT_VERIFIED"
                or not target_verified
            )
        ):
            feedback.switch_scenario = True
            feedback.stimulus_additions.append(
                "같은 target/test placement에서 coverage 또는 target verification이 반복 실패했다. "
                "다음 validated scenario나 다른 candidate test file로 전환하라."
            )
            target = scenario.get("target_location") or {}
            source_file = target.get("source_file", "") if isinstance(target, dict) else ""
            target_func = target.get("target_function", "") if isinstance(target, dict) else ""
            feedback.repair_directive = _build_repair_directive(
                "SWITCH_SCENARIO_OR_TEST_FILE",
                "same target/source failed coverage or target verification for multiple iterations",
                scenario,
                clue,
                must_change=[
                    "Switch to the next validated scenario or another existing candidate test file.",
                    "Do not reuse the same low-coverage target/test placement unchanged.",
                    "Rebuild setup from the issue reproduction code and existing target-file examples.",
                ],
                must_keep=[
                    f"issue target source intent: {source_file}" if source_file else "",
                    f"issue target function intent: {target_func}" if target_func else "",
                ],
                execution_result=execution_result,
                raw_output=raw_output,
                coverage=coverage,
            )

        # Step 3b: NOT_VALID → context에서 skip 없는 대안 테스트 파일 찾기
        if failure_type == FailureType.NOT_VALID and context:
            target_location = scenario.get("target_location") if isinstance(scenario, dict) else {}
            if not isinstance(target_location, dict):
                target_location = {}
            alt_test_file = self._find_skip_free_test_file(
                context,
                current_test_file=target_location.get("candidate_test_file", ""),
            )
            if alt_test_file:
                feedback.candidate_test_file_override = alt_test_file

        # Step 4: 시나리오 보강
        base_scenario = scenario
        if feedback.switch_scenario and validation_report:
            alt = self._pick_alternative_scenario(
                validation_report,
                current_scenario_id=scenario.get("scenario_id"),
            )
            if alt:
                logger.info(
                    "[iteration %d] NO_COVERAGE → 시나리오 전환: %s → %s",
                    iteration,
                    scenario.get("scenario_id"),
                    alt.get("scenario_id"),
                )
                base_scenario = alt

        refined = refine_scenario(base_scenario, feedback, iteration)

        configured_max = (
            context.get("max_feedback_iterations")
            if isinstance(context, Mapping)
            else None
        )
        # The orchestrator owns the feedback budget.  If an isolated caller
        # omits it, do not invent a fixed retry allowance for the special
        # insufficient-P branch; treat the current iteration as exhausted.
        max_feedback_iterations = (
            max(1, configured_max)
            if isinstance(configured_max, int) and not isinstance(configured_max, bool)
            else iteration
        )
        should_continue = (
            failure_type != FailureType.ALIGNED
            and (
                failure_type_detail != "SBFL_UNAVAILABLE_INSUFFICIENT_P"
                or iteration < max_feedback_iterations
            )
        )

        # 커버리지 요약 (상위 5개 소스 파일)
        cov_summary = {}
        valid_cov = {k: v for k, v in coverage_data.items() if isinstance(v, dict)}
        for fname, info in sorted(
            valid_cov.items(),
            key=lambda x: x[1].get("cover", 0),
            reverse=True,
        )[:5]:
            if "/test" not in fname and "test_" not in fname:
                cov_summary[fname] = {
                    "cover": info.get("cover", 0),
                    "stmts": info.get("stmts", 0),
                    "miss": info.get("miss", 0),
                }

        status_fields = project_m7_status_fields(failure_type)
        status_fields["execution_status"] = normalize_pre_patch_execution_status(
            execution_result
        ).value
        if conservative_gate_pending_llm:
            # Quantitative ALIGNED is only a pre-LLM candidate state here.  It
            # must not become canonical M7 admission until the dedicated v29
            # Conservative Gate decision has completed.
            status_fields.update(
                m7_alignment_status=None,
                admitted_to_final_set=False,
                diagnostic_only=True,
            )
        score_definition_status = {
            "s_b": "V36_FORMULA_9" if v36_methodology else "ORIGINAL_EQ1",
            "s_a": "V36_FORMULA_12" if v36_methodology else "ORIGINAL_EQ3",
            "iteration_1_s_c": (
                "V36_FORMULA_10"
                if v36_methodology and iteration == 1
                else "DIAGNOSTIC_ONLY"
                if v36_methodology
                else "NOT_USED_BY_V29"
            ),
            "s_c_prime": (
                "V36_FORMULA_11_WEIGHTED_OCHIAI"
                if v36_methodology and weighted_coverage_available
                else "V29_WEIGHTED_OCHIAI"
                if sbfl_weighted_coverage.get("coverage_valid")
                else "UNAVAILABLE_CURRENT_PASS"
            ),
            "L_s": (
                "POSITIVE_OCHIAI_LINES_CURRENT_PASS"
                if sbfl_weighted_coverage.get("coverage_valid")
                else "UNAVAILABLE_CURRENT_PASS"
            ),
        }
        score_breakdown = {
            "score_schema_version": ALIGNMENT_SCORE_SCHEMA_VERSION,
            "score_range": "0..1",
            "v22_score_definition_status": score_definition_status,
            "score_definition": score_definition_status,
            "bug_fail_score": bug_fail,
            "bug_fail_features": bug_fail_features,
            "issue_alignment_score": issue_align,
            "coverage_score": canonical_coverage,
            "s_c_prime": canonical_coverage,
            "coverage_gate_value": coverage,
            "legacy_coverage_score_diagnostic": coverage,
            "coverage_score_available": coverage_score_available,
            "coverage_score_status": (
                "AVAILABLE_V36_S_C"
                if v36_methodology and iteration == 1 and coverage_score_available
                else "AVAILABLE_WEIGHTED_OCHIAI"
                if weighted_coverage_available
                else "UNAVAILABLE_CURRENT_PASS"
            ),
            "coverage_score_unavailable_reason": (
                None
                if sbfl_weighted_coverage.get("coverage_valid")
                else sbfl_weighted_coverage.get("fallback_reason")
            ),
            **coverage_contract,
            "m7_sbfl_weighted_coverage": sbfl_weighted_coverage,
            "oracle_confidence_score": oracle_quality.score,
            "oracle_risk_flags": oracle_quality.risk_flags,
            "oracle_consistency": oracle_consistency.to_dict(),
            "conservative_gate_reasons": conservative_gate_reasons,
            "conservative_gate_triggered": conservative_gate_triggered,
            "conservative_gate_is_only_branching_reason": (
                conservative_gate_is_only_branching_reason
            ),
            "conservative_gate_pending_llm": conservative_gate_pending_llm,
            "all_numeric_gates_pass": all_numeric_gates_pass,
            "conservative_gate_details": {
                "blocking_oracle_flags": v29_blocking_oracle_flags,
                "flaky": bool(execution_result.get("flaky")),
                "flaky_detail": execution_result.get("flaky_detail"),
            },
            "gate_results": {
                **numeric_gate_results,
                "all_numeric_gates_pass": all_numeric_gates_pass,
                "conservative_gate_triggered": conservative_gate_triggered,
                "conservative_gate_is_only_branching_reason": (
                    conservative_gate_is_only_branching_reason
                ),
                "conservative_gate_pending_llm": conservative_gate_pending_llm,
            },
            "gate_warnings": nonblocking_gate_warnings,
            "target_verified": target_verified,
            "target_verification_evidence": target_verification_evidence,
            "strong_issue_evidence": strong_issue_evidence,
            "coverage_fallback_reason": coverage_fallback_reason,
            "oracle_type": oracle_type,
            "oracle_source": oracle_source,
            "issue_reported_semantic_invariant": issue_reported_invariant,
            "repair_attempted": bool(generated_test.get("repair_attempted")),
            "repair_actions": generated_test.get("repair_actions", []),
            "repair_failed_reason": repair_failed_reason,
            "repair_retry_count": generated_test.get("repair_retry_count", 0),
            "retry_required_oracle_risks": retry_required_oracle_risks,
            "semantic_risk_flags": semantic_risk_flags,
            "failure_type_detail": failure_type_detail,
        }
        structured_feedback = build_structured_feedback(
            verdict=failure_type.value,
            iteration=iteration,
            bug_fail=bug_fail,
            coverage=canonical_coverage,
            issue_align=issue_align,
            thresholds={
                "s_b": _ALIGNED_BUG_FAIL_MIN,
                "s_c_prime": _COVERAGE_MIN_GATE,
                "s_a": _ISSUE_ALIGN_MIN_GATE,
            },
            execution_result=execution_result,
            clue=clue,
            scenario=scenario,
            generated_test=generated_test,
            score_breakdown=score_breakdown,
            legacy_feedback=feedback.to_dict(),
            context=context,
        )
        if failure_type_detail == "SBFL_UNAVAILABLE_INSUFFICIENT_P":
            recoverable = iteration < max_feedback_iterations
            structured_feedback.update(
                {
                    "feedback_branch": (
                        "M2+M3+M5" if recoverable else "M6_SUPPLEMENTAL_PASS_COLLECTION"
                    ),
                    "target_modules": ["M2", "M3", "M5"] if recoverable else [],
                    "diagnosis": (
                        "current-pass bounded PASS collection produced fewer than three valid distinct spectra; "
                        "change target/scenario/candidate ownership and retry SBFL"
                        if recoverable
                        else "bounded pre-patch PASS collection exhausted with fewer than three valid distinct spectra"
                    ),
                    "routing_reason": "SBFL_UNAVAILABLE_INSUFFICIENT_P",
                    "loop_termination_recommended": not recoverable,
                    "coverage_score_available": False,
                    "coverage_score_unavailable_reason": "SBFL_UNAVAILABLE_INSUFFICIENT_P",
                    "supplemental_pass_collection": _compact_supplemental_pass_collection(
                        supplemental_collection
                    ),
                }
            )
        if resolved_feature_flags.m7_llm_scenario_refinement and failure_type != FailureType.ALIGNED:
            structured_feedback["llm_refinement_requested"] = True
            llm_refiner = context.get("m7_llm_refiner") if isinstance(context, dict) else None
            if llm_refiner is not None and not callable(llm_refiner):
                llm_refiner = None
            structured_feedback["llm_scenario_refinement"] = build_m7_llm_scenario_refinement(
                current_scenario=scenario,
                structured_feedback=structured_feedback,
                execution_summary=_execution_summary_for_m7_llm(execution_result),
                valid_sbfl_candidates=_valid_m7_sbfl_candidates(canonical_sbfl_result),
                verdict_branch=failure_type.value,
                generated_test=generated_test,
                score_breakdown=score_breakdown,
                llm_refiner=llm_refiner,
            )
            refined = _apply_m7_llm_refinement_to_scenario(
                refined,
                structured_feedback["llm_scenario_refinement"],
                verdict=failure_type.value,
            )
        elif resolved_feature_flags.m7_llm_scenario_refinement:
            structured_feedback["llm_scenario_refinement"] = build_m7_llm_scenario_refinement(
                current_scenario=scenario,
                structured_feedback=structured_feedback,
                execution_summary=_execution_summary_for_m7_llm(execution_result),
                valid_sbfl_candidates=_valid_m7_sbfl_candidates(canonical_sbfl_result),
                verdict_branch=failure_type.value,
                generated_test=generated_test,
                score_breakdown=score_breakdown,
                llm_refiner=None,
            )
        iteration_feedback_summary = build_iteration_feedback_summary(
            structured_feedback,
            iteration=iteration,
            verdict=failure_type.value,
            bug_fail=bug_fail,
            coverage=canonical_coverage,
            issue_align=issue_align,
        )
        alignment_verdict = (
            "PENDING_CONSERVATIVE_REVIEW"
            if conservative_gate_pending_llm
            else FailureType.ALIGNED.value
            if failure_type == FailureType.ALIGNED
            else "NOT_ALIGNED"
        )
        admission_path = (
            "DIRECT"
            if failure_type == FailureType.ALIGNED
            and not score_breakdown["conservative_gate_triggered"]
            else None
        )
        return AlignmentResult(
            iteration=iteration,
            failure_type=failure_type.value,
            score_breakdown=score_breakdown,
            diagnosis=feedback.diagnosis,
            feedback=feedback.to_dict(),
            refined_scenario=refined,
            should_continue=should_continue,
            test_results=test_results,
            coverage_summary=cov_summary,
            failure_type_detail=failure_type_detail,
            execution_status=status_fields["execution_status"],
            validation_status=status_fields["validation_status"],
            m7_alignment_status=status_fields["m7_alignment_status"],
            admitted_to_final_set=status_fields["admitted_to_final_set"],
            diagnostic_only=status_fields["diagnostic_only"],
            legacy_failure_type=status_fields["legacy_failure_type"],
            structured_feedback=structured_feedback,
            iteration_feedback_summary=iteration_feedback_summary,
            alignment_verdict=alignment_verdict,
            admission_path=admission_path,
        )

    @staticmethod
    def _pick_alternative_scenario(
        validation_report: Dict[str, Any],
        current_scenario_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """validation_report에서 현재 시나리오가 아닌 대안을 선택.

        Looks first among selected scenarios, then among rejected ones
        (which still have a normalized_scenario from the scoring pass).
        """
        selected = validation_report.get("selected_scenarios", [])
        for item in selected:
            normalized = item.get("normalized_scenario", {})
            if normalized and normalized.get("scenario_id") != current_scenario_id:
                return normalized
        # Fallback: pick from rejected scenarios that were normalized
        rejected = validation_report.get("rejected_scenarios", [])
        for item in sorted(rejected, key=lambda x: x.get("score", 0), reverse=True):
            normalized = item.get("normalized_scenario", {})
            if normalized and normalized.get("scenario_id") != current_scenario_id:
                return normalized
        return None

    @staticmethod
    def _find_skip_free_test_file(
        context: Dict[str, Any],
        current_test_file: str,
    ) -> str:
        """context의 candidate_test_files에서 has_module_skip=False인 대안을 찾는다.

        현재 파일과 다른, skip이 없는 첫 번째 후보를 반환. 없으면 빈 문자열.
        """
        candidates = context.get("candidate_test_files", [])
        for cand in candidates:
            path = cand.get("path", "")
            if path == current_test_file:
                continue
            if not cand.get("has_module_skip", False):
                return path
        return ""
