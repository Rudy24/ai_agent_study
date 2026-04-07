"""
5条HR问题RAGAS验证脚本
======================
用法: python ragas/eval_5_questions.py

流程:
1. 从 ragas/data/hr_eval_questions.json 加载前5个问题
2. 对每个问题运行RAG获取答案和上下文
3. 运行RAGAS评估
4. 输出各项指标
"""
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
_boot.ensure_dirs()

# 设置编码（Windows兼容）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 导入项目模块
from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    LOCAL_MODEL_PATH,
    get_ragas_answer_relevancy,
    get_ragas_evaluator_llm,
)
from knowledge_base import KnowledgeBase
from rag_engine import RAGEngine

# 导入RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    from datasets import Dataset
    import pandas as pd
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)


def load_5_questions() -> List[Dict]:
    """加载前5个测试问题"""
    with open(DEFAULT_QUESTIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"][:5]
    print(f"[INFO] 加载了 {len(questions)} 个测试问题")
    return questions


def run_rag_on_questions(questions: List[Dict], engine: RAGEngine) -> List[Dict]:
    """运行RAG获取答案"""
    results = []
    print(f"\n[INFO] 开始运行RAG（共{len(questions)}个问题）...")
    print("-" * 70)

    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"  [{i}/5] {question[:30]}...", end=" ", flush=True)
        try:
            result = engine.ask(question)
            answer = result.get("result", "")
            contexts = [d.page_content for d in result.get("source_documents", [])]

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
            })
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": "",
                "contexts": [],
            })

    print("-" * 70)
    return results


def run_ragas_evaluation(eval_data: List[Dict]) -> pd.DataFrame:
    """运行RAGAS评估"""
    # 准备数据集
    for item in eval_data:
        item.setdefault("contexts", [])
        item.setdefault("ground_truth", item.get("answer", ""))

    dataset = Dataset.from_list(eval_data)

    # 设置评估器
    print("\n[INFO] 初始化RAGAS评估器（DeepSeek n=1）...")
    llm = get_ragas_evaluator_llm()

    model_path = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else EMBEDDING_MODEL_NAME
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # 定义指标
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        get_ragas_answer_relevancy(),
    ]

    print(f"[INFO] 评估指标: {[m.name for m in metrics]}")
    print(f"[INFO] 开始评估 {len(dataset)} 条样本...")
    print("[WARNING] 需要调用 DeepSeek API，请耐心等待...\n")

    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return result.to_pandas()


def show_results(df: pd.DataFrame, eval_data: List[Dict]):
    """显示评估结果（使用ASCII字符避免编码问题）"""
    numeric_cols = [
        c for c in df.columns
        if c not in ["question", "answer", "contexts", "ground_truth"]
    ]

    print("\n" + "=" * 70)
    print("              RAGAS 评估结果 - 5条HR问题")
    print("=" * 70)

    # 显示每个问题的详细结果
    for i, row in df.iterrows():
        print(f"\n【问题 {i+1}】{eval_data[i]['question']}")
        print(f"标准答案: {eval_data[i]['ground_truth'][:50]}...")
        print(f"AI回答: {eval_data[i]['answer'][:80]}...")
        print("-" * 50)
        for col in numeric_cols:
            score = row[col]
            # 确保是单一数值
            if hasattr(score, '__len__') and not isinstance(score, str):
                score = score[0] if len(score) > 0 else None
            # 使用ASCII字符显示进度条
            if pd.notna(score) and isinstance(score, (int, float)):
                try:
                    filled = int(float(score) * 20)
                    bar = "#" * filled + "-" * (20 - filled)
                    print(f"  {col:20s}: {score:.3f} [{bar}]")
                except:
                    print(f"  {col:20s}: {score}")
            else:
                print(f"  {col:20s}: N/A")

    # 显示平均分
    print("\n" + "=" * 70)
    print("                    平均分汇总")
    print("=" * 70)
    for col in numeric_cols:
        try:
            # 尝试转换为数值并计算平均
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            avg = numeric_series.mean()
            if pd.notna(avg):
                filled = int(avg * 20)
                bar = "#" * filled + "-" * (20 - filled)
                print(f"  {col:20s}: {avg:.3f} [{bar}]")
            else:
                print(f"  {col:20s}: N/A")
        except Exception as e:
            print(f"  {col:20s}: Error ({e})")

    # 保存结果
    try:
        out_json = RESULTS_DIR / "eval_5_results.json"
        df.to_json(out_json, orient="records", force_ascii=False, indent=2)
        print(f"\n[INFO] 详细结果已保存: {out_json}")
    except Exception as e:
        print(f"\n[WARN] 保存结果失败: {e}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("        HR制度RAG - 5条数据验证")
    print("=" * 70)

    # 加载5个问题
    questions = load_5_questions()

    # 显示问题列表
    print("\n测试问题:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['question']}")

    # 加载知识库
    print("\n[INFO] 加载知识库...")
    kb = KnowledgeBase()

    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = kb.load_vector_store()
        print("[OK] 索引加载成功")
    else:
        print(f"[ERROR] 索引不存在: {FAISS_INDEX_PATH}")
        sys.exit(1)

    # 创建RAG引擎
    engine = RAGEngine(vectorstore)

    # 运行RAG
    eval_data = run_rag_on_questions(questions, engine)

    # 保存中间结果到 ragas/results/
    ds_path = RESULTS_DIR / "eval_5_dataset.json"
    with open(ds_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 中间数据集: {ds_path}")

    # 运行RAGAS评估
    df = run_ragas_evaluation(eval_data)

    # 显示结果
    show_results(df, eval_data)

    print("\n[OK] 评估完成!")


if __name__ == "__main__":
    main()
