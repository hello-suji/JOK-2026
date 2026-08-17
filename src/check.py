from src.models.client import LLMClient
from src.models.config import load_model_config
import json
import re


def extract_json(raw: str):
    raw = raw.strip()

    # ```json ... ``` 코드블록 제거
    if raw.startswith("```"):
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"^```\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


if __name__ == "__main__":
    model_cfg = load_model_config("qwen")
    client = LLMClient(model_cfg)

    prompt = """
다음 이슈 설명을 읽고 반드시 JSON 객체 하나만 출력하라.
설명, 코드블록, 마크다운, 추가 문장은 절대 쓰지 마라.

출력 형식:
{
  "scenario_id": "S1",
  "failure_hypothesis": "",
  "oracle": ""
}

규칙:
- oracle은 실행 시 판정 가능한 문장으로 작성할 것
- '검토한다', '확인한다', '해결한다' 같은 추상 표현 금지
- oracle은 기대 반환값, 기대 예외, 기대 상태 변화 중 하나로 작성할 것

이슈:
Cannot override get_FOO_display() in Django 2.2+
Expected:
Overridden get_foo_bar_display() should return "something"
Actual:
Default choice label is returned instead
"""

    raw = client.generate(prompt)
    print("원본 응답:\n", raw)

    parsed = extract_json(raw)
    print("\n파싱 결과:")
    print(parsed)