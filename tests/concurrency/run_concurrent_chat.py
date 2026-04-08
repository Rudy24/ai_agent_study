# -*- coding: utf-8 -*-
"""
并发压测：对 HR RAG 的 POST /api/chat 发起多路同时请求，统计成功率与延迟分位数。
依赖：仅 Python 标准库（urllib + asyncio）；请先启动 api_server。

用法示例:
  python tests/concurrency/run_concurrent_chat.py
  python tests/concurrency/run_concurrent_chat.py --url http://127.0.0.1:8000/api/chat -c 10 -n 30
  python tests/concurrency/run_concurrent_chat.py --api-key your-secret --timeout 180

环境变量（可选）:
  CHAT_STRESS_URL      覆盖默认 http://127.0.0.1:8000/api/chat
  CHAT_STRESS_API_KEY  与请求头 X-API-Key 一致（同 API_SERVICE_API_KEY）
"""
from __future__ import annotations

import argparse  # 命令行
import asyncio  # 异步调度与信号量限流
import json  # 请求体与响应解析
import os  # 读环境变量
import ssl  # HTTPS 默认校验
import sys  # 退出码
import time  # 计时
import urllib.error  # HTTP 错误
import urllib.request  # 同步 HTTP 客户端（在线程中执行）
from typing import Any, Dict, List, Optional, Tuple


# 默认与 README / 典型验收问题一致，可 --questions-file 换行分隔自定义
DEFAULT_QUESTIONS: List[str] = [
    "每人每月有多少次免费补卡机会？",
    "迟到31分钟到60分钟以内，负激励多少钱？",
    "旷工当日的工资如何计算？另有什么处罚？",
]


def _load_questions(path: Optional[str]) -> List[str]:
    """从文件加载问题列表（UTF-8，每行一条）；空行跳过。"""
    if not path:
        return list(DEFAULT_QUESTIONS)
    out: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out if out else list(DEFAULT_QUESTIONS)


def _post_chat_sync(
    url: str,
    question: str,
    api_key: Optional[str],
    timeout_sec: float,
) -> Tuple[int, Dict[str, Any]]:
    """同步 POST /api/chat，返回 (HTTP 状态码, 解析后的 JSON 对象)。"""
    payload = json.dumps(
        {"question": question, "conversation_id": None},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout_sec, context=ctx) as resp:
        raw = resp.read().decode("utf-8")
        code = resp.getcode()
        return code, json.loads(raw) if raw.strip() else {}


def _percentile(sorted_vals: List[float], p: float) -> float:
    """线性插值分位数；sorted_vals 已升序。"""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def _one_request(
    sem: asyncio.Semaphore,
    url: str,
    question: str,
    api_key: Optional[str],
    timeout_sec: float,
) -> Dict[str, Any]:
    """单请求：信号量限流 + 线程池跑同步 urllib，避免阻塞事件循环。"""
    async with sem:
        t0 = time.perf_counter()
        try:
            code, body = await asyncio.to_thread(
                _post_chat_sync, url, question, api_key, timeout_sec
            )
            dt = time.perf_counter() - t0
            ok = 200 <= code < 300 and bool(
                (body.get("result") or body.get("answer_only") or "").strip()
            )
            return {
                "ok": ok,
                "latency_sec": dt,
                "status": code,
                "err": None,
            }
        except urllib.error.HTTPError as e:
            dt = time.perf_counter() - t0
            return {
                "ok": False,
                "latency_sec": dt,
                "status": e.code,
                "err": e.read().decode("utf-8", errors="replace")[:500],
            }
        except Exception as e:
            dt = time.perf_counter() - t0
            return {
                "ok": False,
                "latency_sec": dt,
                "status": None,
                "err": str(e)[:500],
            }


async def run_stress(
    url: str,
    concurrency: int,
    total: int,
    questions: List[str],
    api_key: Optional[str],
    timeout_sec: float,
) -> List[Dict[str, Any]]:
    """构造 total 个任务，同一时刻最多 concurrency 个在飞。"""
    sem = asyncio.Semaphore(max(1, concurrency))
    tasks: List[asyncio.Task] = []
    for i in range(total):
        q = questions[i % len(questions)]
        tasks.append(
            asyncio.create_task(_one_request(sem, url, q, api_key, timeout_sec))
        )
    return list(await asyncio.gather(*tasks))


def _print_report(rows: List[Dict[str, Any]]) -> int:
    """打印汇总；返回进程退出码（有失败则为 1）。"""
    n = len(rows)
    oks = [r for r in rows if r["ok"]]
    fails = [r for r in rows if not r["ok"]]
    lats = sorted(r["latency_sec"] for r in oks)
    print("=" * 60)
    print("  HR RAG /api/chat 并发压测结果")
    print("=" * 60)
    print(f"  总请求数     : {n}")
    print(f"  成功         : {len(oks)}")
    print(f"  失败         : {len(fails)}")
    if lats:
        print(f"  延迟 avg (s) : {sum(lats) / len(lats):.3f}")
        print(f"  延迟 min (s) : {lats[0]:.3f}")
        print(f"  延迟 max (s) : {lats[-1]:.3f}")
        print(f"  延迟 p50 (s) : {_percentile(lats, 50):.3f}")
        print(f"  延迟 p95 (s) : {_percentile(lats, 95):.3f}")
    if fails:
        print("-" * 60)
        print("  失败样例（最多 5 条）:")
        for r in fails[:5]:
            print(f"    status={r['status']} err={r['err']}")
    print("=" * 60)
    return 0 if not fails else 1


def main() -> int:
    """解析参数并运行 asyncio 压测。"""
    default_url = os.getenv("CHAT_STRESS_URL", "http://127.0.0.1:8000/api/chat").strip()
    default_key = os.getenv("CHAT_STRESS_API_KEY", "").strip() or None

    p = argparse.ArgumentParser(description="对 /api/chat 做并发压测（标准库）")
    p.add_argument("--url", default=default_url, help="完整 chat 接口 URL")
    p.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="同时进行的请求数上限",
    )
    p.add_argument("-n", "--total", type=int, default=15, help="总请求次数")
    p.add_argument(
        "--questions-file",
        default=None,
        help="问题列表文件 UTF-8 每行一条；默认使用内置 3 条 HR 题",
    )
    p.add_argument("--api-key", default=default_key, help="X-API-Key，不配则不带")
    p.add_argument("--timeout", type=float, default=120.0, help="单请求超时秒数")
    args = p.parse_args()

    qs = _load_questions(args.questions_file)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    rows = asyncio.run(
        run_stress(
            args.url,
            args.concurrency,
            args.total,
            qs,
            args.api_key,
            args.timeout,
        )
    )
    return _print_report(rows)


if __name__ == "__main__":
    raise SystemExit(main())
