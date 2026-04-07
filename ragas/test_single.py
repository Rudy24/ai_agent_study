# -*- coding: utf-8 -*-
"""
单条问题 RAG + RAGAS 快速验证。
用法:
  python ragas/test_single.py
  python ragas/test_single.py -q "年假有多少天？"
"""
import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_ragas_workspace_bootstrap",
    Path(__file__).resolve().parent / "bootstrap.py",
)
_boot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_boot)
DEFAULT_QUESTIONS_JSON = _boot.DEFAULT_QUESTIONS_JSON
RESULTS_DIR = _boot.RESULTS_DIR
_boot.ensure_dirs()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    LOCAL_MODEL_PATH,
    get_ragas_answer_relevancy,
    get_ragas_evaluator_llm,
)
from knowledge_base import KnowledgeBase  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402

try:
    from datasets import Dataset  # noqa: E402
    from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
    import pandas as pd  # noqa: E402
    from ragas import evaluate  # noqa: E402
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
    from ragas.llms import LangchainLLMWrapper  # noqa: E402
    from ragas.metrics import (  # noqa: E402
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)


def run_ragas_one(row: Dict[str, Any]) -> pd.DataFrame:
    """对单条样本跑 RAGAS。"""
    for k in ("contexts", "ground_truth"):
        row.setdefault(k, row.get("answer", "") if k == "ground_truth" else [])
    ds = Dataset.from_list([row])
    llm = get_ragas_evaluator_llm()
    mp = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else EMBEDDING_MODEL_NAME
    emb = HuggingFaceEmbeddings(
        model_name=mp,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    mets = [context_precision, context_recall, faithfulness, get_ragas_answer_relevancy()]
    res = evaluate(
        dataset=ds,
        metrics=mets,
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(emb),
    )
    return res.to_pandas()


def main():
    """解析参数，跑一条 RAG 与 RAGAS，并写入 results。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", type=str, default=None, help="问题文本")
    ap.add_argument("-t", "--truth", type=str, default=None, help="标准答案（可选）")
    args = ap.parse_args()

    q = args.question
    gt = args.truth
    if not q:
        with open(DEFAULT_QUESTIONS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        first = data["questions"][0]
        q = first["question"]
        gt = first.get("ground_truth", "")

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"[ERROR] 无索引: {FAISS_INDEX_PATH}")
        sys.exit(1)

    kb = KnowledgeBase()
    vs = kb.load_vector_store()
    eng = RAGEngine(vs)
    print(f"[User] {q}")
    r = eng.ask(q)
    answer = r.get("result", "")
    ctxs: List[str] = [d.page_content for d in r.get("source_documents", [])]
    print(f"[AI] {answer[:500]}...")

    row = {"question": q, "ground_truth": gt or "", "answer": answer, "contexts": ctxs}
    df = run_ragas_one(row)

    for col in df.columns:
        if col in ("question", "answer", "contexts", "ground_truth"):
            continue
        v = df.iloc[0][col]
        if hasattr(v, "__len__") and not isinstance(v, str):
            v = v[0] if len(v) else None
        if pd.notna(v) and isinstance(v, (int, float)):
            print(f"  {col}: {float(v):.3f}")

    out_base = RESULTS_DIR / "test_single"
    df.to_json(f"{out_base}.json", orient="records", force_ascii=False, indent=2)
    with open(f"{out_base}_dataset.json", "w", encoding="utf-8") as f:
        json.dump([row], f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入 {out_base}.json")


if __name__ == "__main__":
    main()
