"""
Score government sentiment toward predefined industries in Xinwen Lianbo transcripts using GPT-5 mini.

Prereqs:
- .env with OPENAI_API_KEY
- uv or pip: uv pip install openai python-dotenv
Run:
  uv run analyze_industry_sentiment.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5-mini"
INDUSTRIES_PATH = Path("Industries_unique.txt")
TRANSCRIPTS_DIR = Path("data/xinwenlianbo_texts")
OUTPUT_JSON = Path("data/industry_sentiment.json")


def load_industries(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cache(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_messages(industries: List[str], transcript: str) -> List[Dict[str, str]]:
    industries_block = "\n".join(f"- {name}" for name in industries)
    user_prompt = f"""\
你是政策情绪分析员。给定当日《新闻联播》全文，请对下列每个行业给出政府态度的情绪评分。

评分标准：
- 数值范围：-100 到 100，整数。
- 正数表示积极/利好，负数表示消极/利空，0 表示未提及或无法判断。
- 依据新闻内容对行业未来走势的倾向进行判断。

输出格式：
返回一个 JSON 对象，键为行业名称，值为包含 score 和 rationale 的对象，rationale 用一句简短中文概括依据。
示例：
{{
  "Agriculture": {{"score": 20, "rationale": "提到加大农业补贴，利好农业"}},
  "Banking": {{"score": 0, "rationale": "未提及银行业"}}
}}

行业列表：
{industries_block}

新闻全文：
{transcript}
"""
    return [
        {
            "role": "system",
            "content": "You are a precise Chinese policy sentiment rater. Follow the format exactly and be concise.",
        },
        {"role": "user", "content": user_prompt},
    ]


def call_model(client: OpenAI, messages: List[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def append_results(
    cache: Dict[str, Dict[str, object]], path: Path, date: str, model: str, data: Dict[str, Dict[str, object]]
) -> None:
    cache[date] = {"model": model, "results": data}
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_transcripts(dir_path: Path) -> Iterable[Path]:
    return sorted(p for p in dir_path.glob("*.txt"))


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in environment (.env).")

    industries = load_industries(INDUSTRIES_PATH)
    cache = load_cache(OUTPUT_JSON)
    done_dates = set(cache.keys())
    client = OpenAI()

    for transcript_path in iter_transcripts(TRANSCRIPTS_DIR):
        date = transcript_path.stem
        if date in done_dates:
            continue

        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            continue

        messages = build_messages(industries, transcript)
        print(f"Scoring {date} with {MODEL}...")
        result = call_model(client, messages)
        append_results(cache, OUTPUT_JSON, date, MODEL, result)


if __name__ == "__main__":
    main()
