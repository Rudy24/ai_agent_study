"""
快速评估脚本（优化版）- 不对比原版
===================================
结果保存到 ragas/results/ 目录

用法:
  python ragas/eval_optimized_quick.py --num 5
  python ragas/eval_optimized_quick.py --all
"""
import importlib.util
import argparse
import io
import json
import os
import sys
from datetime import datetime
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
RESULTS_DIR = _boot.RESULTS_DIR
DEFAULT_QUESTIONS_JSON = _boot.DEFAULT_QUESTIONS_JSON
_boot.ensure_dirs()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    LOCAL_MODEL_PATH,
    get_ragas_answer_relevancy,
    get_ragas_evaluator_llm,
)
from knowledge_base import KnowledgeBase
from rag_engine_optimized import RAGEngineOptimized

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


def load_questions(input_file: str, num: int = None) -> List[Dict]:
    """加载测试问题"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data["questions"]
    
    if num and num > 0:
        actual_num = min(num, len(questions))
        questions = questions[:actual_num]
    
    print(f"[INFO] 从 {input_file} 加载了 {len(questions)} 个测试问题")
    return questions


def run_rag(questions: List[Dict], engine) -> List[Dict]:
    """批量运行RAG获取答案"""
    results = []
    total = len(questions)
    
    print(f"\n{'='*70}")
    print(f"运行优化版RAG引擎（共 {total} 个问题）")
    print(f"{'='*70}")

    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"\n[{i}/{total}] {question}")
        print("-" * 50)
        
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
            print(f"[OK] 回答: {answer[:80]}...")
            print(f"[INFO] 引用 {len(contexts)} 个文档")
        except Exception as e:
            print(f"[FAIL] {str(e)[:100]}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": "",
                "contexts": [],
            })

    success_count = len([r for r in results if r["answer"]])
    print(f"\n[SUMMARY] 成功处理 {success_count}/{total} 条")
    return results


def run_ragas(eval_data: List[Dict]) -> pd.DataFrame:
    """运行RAGAS评估"""
    for item in eval_data:
        item.setdefault("contexts", [])
        item.setdefault("ground_truth", item.get("answer", ""))

    dataset = Dataset.from_list(eval_data)

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

    metrics = [context_precision, context_recall, faithfulness, get_ragas_answer_relevancy()]

    print(f"[INFO] 开始评估 {len(dataset)} 条样本...")
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return result.to_pandas()


def display_results(df: pd.DataFrame, eval_data: List[Dict]):
    """显示评估结果"""
    scores = {}
    
    # 计算平均分
    score_cols = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    for col in score_cols:
        try:
            values = []
            for val in df[col]:
                if hasattr(val, '__len__') and not isinstance(val, str):
                    val = val[0] if len(val) > 0 else None
                if pd.notna(val) and isinstance(val, (int, float)):
                    values.append(float(val))
            
            if values:
                scores[col] = sum(values) / len(values)
        except:
            pass
    
    print(f"\n{'='*70}")
    print("                    评估结果")
    print(f"{'='*70}")
    
    # 每个问题的结果
    for i, row in df.iterrows():
        print(f"\n【问题 {i+1}】{eval_data[i]['question']}")
        
        for col in score_cols:
            val = row.get(col)
            if hasattr(val, '__len__') and not isinstance(val, str):
                val = val[0] if len(val) > 0 else None
            
            if pd.notna(val) and isinstance(val, (int, float)):
                bar_filled = int(float(val) * 20)
                bar = "#" * bar_filled + "-" * (20 - bar_filled)
                print(f"  {col:20s}: {val:.3f} [{bar}]")
    
    # 平均分汇总
    print(f"\n{'='*70}")
    print("                    平均分汇总")
    print(f"{'='*70}")
    
    total_score = 0
    valid_count = 0
    
    for col, avg in scores.items():
        bar_filled = int(avg * 20)
        bar = "#" * bar_filled + "-" * (20 - bar_filled)
        print(f"  {col:20s}: {avg:.3f} [{bar}]")
        total_score += avg
        valid_count += 1
    
    if valid_count > 0:
        overall = total_score / valid_count
        print(f"{'-'*70}")
        print(f"  {'综合得分':20s}: {overall:.3f}")
    
    return scores


def save_results(df: pd.DataFrame, eval_data: List[Dict], output_dir: str):
    """保存结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存Excel
    excel_file = os.path.join(output_dir, f"optimized_{timestamp}.xlsx")
    try:
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"\n[OK] Excel已保存: {excel_file}")
    except Exception as e:
        print(f"[WARN] Excel保存失败: {e}")
    
    # 保存JSON
    json_file = os.path.join(output_dir, f"optimized_{timestamp}.json")
    try:
        df.to_json(json_file, orient="records", force_ascii=False, indent=2)
        print(f"[OK] JSON已保存: {json_file}")
    except Exception as e:
        print(f"[WARN] JSON保存失败: {e}")
    
    # 保存数据集
    dataset_file = os.path.join(output_dir, f"dataset_{timestamp}.json")
    try:
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 数据集已保存: {dataset_file}")
    except Exception as e:
        print(f"[WARN] 数据集保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="优化版RAG评估")
    parser.add_argument("--num", "-n", type=int, default=5, help="测试问题数量")
    parser.add_argument("--all", "-a", action="store_true", help="测试全部问题")
    args = parser.parse_args()

    print("="*70)
    print("        HR制度RAG - 优化版评估")
    print("="*70)

    # 加载问题
    test_num = None if args.all else args.num
    questions = load_questions(str(DEFAULT_QUESTIONS_JSON), test_num)
    
    if not questions:
        print("[ERROR] 没有加载到问题")
        sys.exit(1)
    
    # 显示问题
    print("\n测试问题:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['question']}")
    
    # 加载知识库
    print("\n[INFO] 加载知识库...")
    kb = KnowledgeBase()
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"[ERROR] 索引不存在: {FAISS_INDEX_PATH}")
        sys.exit(1)
    
    vectorstore = kb.load_vector_store()
    print("[OK] 索引加载成功")
    
    # 运行优化版RAG
    print("\n" + "="*70)
    print("运行优化版RAG引擎")
    print("="*70)
    
    engine = RAGEngineOptimized(vectorstore)
    eval_data = run_rag(questions, engine)
    
    # 运行RAGAS评估
    print("\n" + "="*70)
    print("运行RAGAS评估")
    print("="*70)
    
    df = run_ragas(eval_data)
    scores = display_results(df, eval_data)
    
    # 保存结果
    save_results(df, eval_data, str(RESULTS_DIR))
    
    print("\n" + "="*70)
    print("[OK] 评估完成！结果已保存到 ragas/results/ 目录")
    print("="*70)


if __name__ == "__main__":
    main()
