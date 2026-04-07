# -*- coding: utf-8 -*-
"""
HR 全量（30 问）RAGAS 评估。
用法:
  python ragas/eval_hr_30.py
  python ragas/eval_hr_30.py --skip-rag --dataset ragas/results/hr_30_dataset.json
"""
import argparse
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

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
resolve_results_path = _boot.resolve_results_path
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
    print(f"[ERROR] 依赖导入失败: {e}")
    sys.exit(1)


def load_hr_questions(path: str) -> List[Dict]:
    """从 JSON 加载 questions 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def run_rag_on_questions(questions: List[Dict], engine: RAGEngine) -> List[Dict]:
    """逐条调用 RAG，构造 RAGAS 所需字段。"""
    results = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"  [{i}/{len(questions)}] {q[:40]}...", end=" ", flush=True)
        try:
            r = engine.ask(q)
            results.append(
                {
                    "question": q,
                    "ground_truth": gt,
                    "answer": r.get("result", ""),
                    "contexts": [d.page_content for d in r.get("source_documents", [])],
                }
            )
            print("OK")
        except Exception as ex:
            print(f"FAIL {ex}")
            results.append(
                {
                    "question": q,
                    "ground_truth": gt,
                    "answer": "",
                    "contexts": [],
                }
            )
    return results


def run_ragas(eval_data: List[Dict]) -> pd.DataFrame:
    """对已有 answer/contexts 跑 RAGAS 四指标。"""
    for item in eval_data:
        item.setdefault("contexts", [])
        item.setdefault("ground_truth", item.get("answer", ""))

    dataset = Dataset.from_list(eval_data)
    llm = get_ragas_evaluator_llm()
    model_path = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else EMBEDDING_MODEL_NAME
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    ev_llm = LangchainLLMWrapper(llm)
    ev_emb = LangchainEmbeddingsWrapper(embeddings)
    metrics = [context_precision, context_recall, faithfulness, get_ragas_answer_relevancy()]
    print(f"[INFO] RAGAS 指标: {[m.name for m in metrics]}")
    out = evaluate(dataset=dataset, metrics=metrics, llm=ev_llm, embeddings=ev_emb)
    return out.to_pandas()


def main():
    """入口：可选跳过 RAG 仅评估已保存数据集。"""
    p = argparse.ArgumentParser(description="HR 30 问 RAGAS")
    p.add_argument("--skip-rag", action="store_true", help="不调用 RAG，直接读数据集 JSON")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="与 --skip-rag 配合，默认 ragas/results/hr_30_dataset.json",
    )
    p.add_argument(
        "--output",
        type=str,
        default="hr_30_ragas.xlsx",
        help="结果 Excel（相对路径写入 ragas/results/）",
    )
    args = p.parse_args()

    if args.skip_rag:
        ds_path = args.dataset or str(RESULTS_DIR / "hr_30_dataset.json")
        if not os.path.isfile(ds_path):
            print(f"[ERROR] 找不到数据集: {ds_path}")
            sys.exit(1)
        with open(ds_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"[INFO] 已加载数据集 {len(eval_data)} 条（跳过 RAG）")
    else:
        questions = load_hr_questions(str(DEFAULT_QUESTIONS_JSON))
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"[ERROR] 无索引: {FAISS_INDEX_PATH}")
            sys.exit(1)
        kb = KnowledgeBase()
        vs = kb.load_vector_store()
        engine = RAGEngine(vs)
        print(f"[INFO] 对 {len(questions)} 条运行 RAG...")
        eval_data = run_rag_on_questions(questions, engine)
        ds_out = RESULTS_DIR / "hr_30_dataset.json"
        with open(ds_out, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 数据集已保存: {ds_out}")

    df = run_ragas(eval_data)
    out_xlsx = resolve_results_path(args.output)
    df.to_excel(out_xlsx, index=False, engine="openpyxl")
    print(f"[OK] RAGAS 结果: {out_xlsx}")
    json_path = Path(out_xlsx).with_suffix(".json")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"[OK] JSON: {json_path}")


if __name__ == "__main__":
    main()
