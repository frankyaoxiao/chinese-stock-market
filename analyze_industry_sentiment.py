from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm


MODEL = "gpt-5-mini"
INDUSTRIES_PATH = Path("Industries_unique.txt")
TRANSCRIPTS_DIR = Path("data/xinwenlianbo_texts")
OUTPUT_JSON = Path("data/industry_sentiment.json")
ERROR_DIR = Path("data/industry_sentiment_errors")


def load_industries(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cache(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache: Dict[str, Dict[str, object]], path: Path) -> None:
    # Caller must hold lock.
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def build_messages(industries: List[str], transcript: str) -> List[Dict[str, object]]:
    industries_block = "\n".join(f"- {name}" for name in industries)
    user_prompt = f"""\
你是政策情绪分析员。给定当日《新闻联播》全文，请对下列每个行业给出政府态度的情绪评分。

评分标准：
- 数值范围：-100 到 100，整数。
- 正数表示积极/利好，负数表示消极/利空，0 表示未提及或无法判断。
- 依据新闻内容对行业未来走势的倾向进行判断。

输出格式（必须为一个 JSON 对象，不要输出其他内容）：
键为行业名称，值为包含 score 和 rationale 的对象；rationale 请控制在 20 个汉字以内。
未提及则 score=0，rationale 写“未提及”。

示例：
{{
  "Agriculture": {{"score": 20, "rationale": "加大农业补贴"}},
  "Banking": {{"score": 0, "rationale": "未提及"}}
}}

行业列表：
{industries_block}

新闻全文：
{transcript}
"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a precise Chinese policy sentiment rater. Respond with JSON only."}],
        },
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]


class ResponseError(Exception):
    def __init__(self, message: str, response: Dict[str, object] | None = None, preview: str | None = None):
        super().__init__(message)
        self.response = response
        self.preview = preview


async def call_model(
    client: AsyncOpenAI, messages: List[Dict[str, object]], max_completion_tokens: int = 3000
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, object]]:
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            max_completion_tokens=max_completion_tokens,
        )
        raw = resp.model_dump()
        content_text = (resp.choices[0].message.content or "").strip()
        if not content_text:
            raise ResponseError("empty response content", raw, "")
        return json.loads(content_text), raw
    except json.JSONDecodeError as exc:
        preview = content_text[:200] if "content_text" in locals() else ""
        raise ResponseError("json decode error", raw if "raw" in locals() else None, preview) from exc
    except Exception as exc:
        raise ResponseError(str(exc), raw if "raw" in locals() else None, "") from exc


def log_error(date: str, err: ResponseError) -> None:
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date,
        "error": str(err),
        "content_preview": err.preview,
        "response": err.response,
    }
    (ERROR_DIR / f"{date}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def append_results(
    cache: Dict[str, Dict[str, object]],
    path: Path,
    date: str,
    model: str,
    data: Dict[str, Dict[str, object]],
    lock: asyncio.Lock,
) -> None:
    async with lock:
        cache[date] = {"model": model, "results": data}
        save_cache(cache, path)


def iter_transcripts(dir_path: Path) -> Iterable[Path]:
    return sorted(p for p in dir_path.glob("*.txt"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score industry sentiment for Xinwen Lianbo transcripts.")
    parser.add_argument("--date", help="Run for a specific date (YYYYMMDD).")
    parser.add_argument("--limit", type=int, help="Process at most N unscored days (in order).")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent requests.")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in environment (.env).")

    industries = load_industries(INDUSTRIES_PATH)
    cache = load_cache(OUTPUT_JSON)
    done_dates = set(cache.keys())
    client = AsyncOpenAI()
    lock = asyncio.Lock()

    targets = [TRANSCRIPTS_DIR / f"{args.date}.txt"] if args.date else list(iter_transcripts(TRANSCRIPTS_DIR))

    to_process: List[Tuple[str, str]] = []
    for transcript_path in targets:
        date = transcript_path.stem
        if date in done_dates:
            continue
        transcript = transcript_path.read_text(encoding="utf-8").strip()
        if not transcript:
            continue
        to_process.append((date, transcript))
        if args.limit is not None and len(to_process) >= args.limit:
            break

    if not to_process:
        return

    sem = asyncio.Semaphore(max(1, args.concurrency))
    pbar = tqdm(total=len(to_process), desc="Scoring days")

    async def worker(date: str, transcript: str) -> None:
        async with sem:
            messages = build_messages(industries, transcript)
            try:
                result, _raw = await call_model(client, messages)
            except ResponseError as err:
                log_error(date, err)
            else:
                await append_results(cache, OUTPUT_JSON, date, MODEL, result, lock)
            finally:
                pbar.update(1)

    tasks = [asyncio.create_task(worker(date, transcript)) for date, transcript in to_process]
    await asyncio.gather(*tasks)
    pbar.close()


if __name__ == "__main__":
    asyncio.run(main_async())
