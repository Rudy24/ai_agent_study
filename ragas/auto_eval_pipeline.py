# -*- coding: utf-8 -*-
"""
自动化：从模板抽题 -> RAG 填答案 -> RAGAS 评分 -> 写入 ragas/output。
用法: python ragas/auto_eval_pipeline.py --num 20
"""
import argparse
import importlib.util
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_ragas_workspace_bootstrap",
    Path(__file__).resolve().parent / "bootstrap.py",
)
_boot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_boot)
OUTPUT_DIR = _boot.OUTPUT_DIR
_boot.ensure_dirs()

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
    print(f"[ERROR] {e}")
    sys.exit(1)

# 主题 -> 问题模板（与制度文档常见块对应）
QUESTION_TEMPLATES: Dict[str, List[str]] = {
    "年假": ["员工每年有多少天年假？", "年假如何申请？"],
    "病假": ["请病假需要什么证明？", "病假工资怎么算？"],
    "离职": ["离职需要提前多久申请？", "试用期离职提前几天？"],
    "考勤": ["迟到怎么处理？", "公司实行什么工作制？"],
    "入职": ["入职需要提交哪些资料？", "管理岗最终面试谁定？"],
}


def pick_questions(n: int) -> List[Tuple[str, str]]:
    """随机抽取 n 个 (主题, 问题)。"""
    flat: List[Tuple[str, str]] = []
    for topic, qs in QUESTION_TEMPLATES.items():
        for q in qs:
            flat.append((topic, q))
    random.shuffle(flat)
    return flat[:n]


def run_rag_batch(pairs: List[Tuple[str, str]], engine: RAGEngine) -> List[Dict]:
    """批量 RAG；ground_truth 用首段上下文截断作弱监督。"""
    rows = []
    for topic, q in pairs:
        try:
            r = engine.ask(q)
            ans = r.get("result", "")
            ctxs = [d.page_content for d in r.get("source_documents", [])]
            gt = (ctxs[0][:300] if ctxs else ans[:300])
            rows.append(
                {
                    "question": q,
                    "answer": ans,
                    "contexts": ctxs,
                    "ground_truth": gt,
                    "topic": topic,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "question": q,
                    "answer": "",
                    "contexts": [],
                    "ground_truth": "",
                    "topic": topic,
                    "error": str(e),
                }
            )
    return rows


def run_ragas(eval_data: List[Dict]) -> pd.DataFrame:
    """RAGAS 四指标。"""
    for it in eval_data:
        it.setdefault("contexts", [])
        it.setdefault("ground_truth", it.get("answer", ""))
    ds = Dataset.from_list(eval_data)
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
    """创建一次带时间戳的输出目录并写入数据集与 RAGAS 报表。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", "-n", type=int, default=20, help="抽样问题数")
    args = ap.parse_args()

    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"[ERROR] 无索引: {FAISS_INDEX_PATH}")
        sys.exit(1)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pairs = pick_questions(args.num)
    kb = KnowledgeBase()
    vs = kb.load_vector_store()
    eng = RAGEngine(vs)
    print(f"[INFO] 生成 {len(pairs)} 条 RAG 样本 -> {run_dir}")
    eval_rows = run_rag_batch(pairs, eng)

    ds_path = run_dir / "eval_dataset.json"
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, ensure_ascii=False, indent=2)

    df = run_ragas(eval_rows)
    xlsx_path = run_dir / "ragas_results.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    df.to_json(run_dir / "ragas_results.json", orient="records", force_ascii=False, indent=2)

    print(f"[OK] 数据集: {ds_path}")
    print(f"[OK] RAGAS: {xlsx_path}")
    # 平均分
    for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        if col in df.columns:
            print(f"  avg {col}: {pd.to_numeric(df[col], errors='coerce').mean():.3f}")


if __name__ == "__main__":
    main()
