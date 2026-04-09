"""
N条HR问题RAGAS验证脚本（优化版）
================================
使用优化的RAG引擎（查询扩展 + 相似度过滤）
结果默认保存到 ragas/results/ 目录

用法:
  python ragas/eval_n_optimized.py --num 5
  python ragas/eval_n_optimized.py --all
  python ragas/eval_n_optimized.py --num 3 --compare

参数:
  --num: 测试问题数量（默认5条）
  --all: 测试全部问题（与 hr_eval_questions.json 条数一致，当前为 50 条）
  --compare: 同时运行原版和优化版进行对比
  --output-dir: 输出目录（默认 ragas/results）
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

# 导入项目模块
from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    LOCAL_MODEL_PATH,
    get_ragas_answer_relevancy,
    get_ragas_evaluator_llm,
)
from knowledge_base import KnowledgeBase

# 导入两个版本的RAG引擎
from rag_engine import RAGEngine as RAGEngineOriginal
from rag_engine_optimized import RAGEngineOptimized

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


def run_rag_batch(questions: List[Dict], engine, engine_name: str) -> List[Dict]:
    """批量运行RAG获取答案"""
    results = []
    total = len(questions)
    
    print(f"\n{'='*70}")
    print(f"使用 {engine_name} 运行RAG（共 {total} 个问题）")
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
            print(f"[OK] 回答: {answer[:100]}...")
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
    print(f"\n[SUMMARY] {engine_name}: 成功处理 {success_count}/{total} 条")
    return results


def run_ragas_evaluation(eval_data: List[Dict]) -> pd.DataFrame:
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

    print(f"[INFO] 评估指标: {[m.name for m in metrics]}")
    print(f"[INFO] 开始评估 {len(dataset)} 条样本...")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return result.to_pandas()


def calculate_average_scores(df: pd.DataFrame) -> Dict[str, float]:
    """计算平均分"""
    score_cols = [c for c in df.columns if c not in 
                  ['question', 'answer', 'contexts', 'ground_truth', 
                   'user_input', 'retrieved_contexts', 'reference', 'response']]
    
    scores = {}
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
    
    return scores


def display_results(df: pd.DataFrame, eval_data: List[Dict], title: str):
    """显示评估结果"""
    scores = calculate_average_scores(df)
    
    print(f"\n{'='*70}")
    print(f"              {title}")
    print(f"{'='*70}")
    
    # 每个问题的结果
    for i, row in df.iterrows():
        print(f"\n【问题 {i+1}】{eval_data[i]['question'][:40]}...")
        
        for col, score in scores.items():
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


def save_results(df: pd.DataFrame, eval_data: List[Dict], output_dir: str, suffix: str):
    """保存结果到指定输出目录（默认 ragas/results）。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存Excel
    excel_file = os.path.join(output_dir, f"eval_{suffix}_{timestamp}.xlsx")
    try:
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"[OK] Excel已保存: {excel_file}")
    except Exception as e:
        print(f"[WARN] Excel保存失败: {e}")
    
    # 保存JSON
    json_file = os.path.join(output_dir, f"eval_{suffix}_{timestamp}.json")
    try:
        df.to_json(json_file, orient="records", force_ascii=False, indent=2)
        print(f"[OK] JSON已保存: {json_file}")
    except Exception as e:
        print(f"[WARN] JSON保存失败: {e}")
    
    # 保存数据集
    dataset_file = os.path.join(output_dir, f"dataset_{suffix}_{timestamp}.json")
    try:
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 数据集已保存: {dataset_file}")
    except Exception as e:
        print(f"[WARN] 数据集保存失败: {e}")
    
    return excel_file, json_file, dataset_file


def compare_versions(scores_original: Dict, scores_optimized: Dict):
    """对比原版和优化版的结果"""
    print(f"\n{'='*70}")
    print("              原版 vs 优化版 对比")
    print(f"{'='*70}")
    print(f"{'指标':<20} {'原版':<15} {'优化版':<15} {'提升':<10}")
    print(f"{'-'*70}")
    
    all_metrics = set(scores_original.keys()) | set(scores_optimized.keys())
    
    for metric in sorted(all_metrics):
        orig = scores_original.get(metric, 0)
        opt = scores_optimized.get(metric, 0)
        improvement = opt - orig
        sign = "+" if improvement >= 0 else ""
        print(f"{metric:<20} {orig:<15.3f} {opt:<15.3f} {sign}{improvement:<9.3f}")
    
    # 计算综合提升
    avg_orig = sum(scores_original.values()) / len(scores_original) if scores_original else 0
    avg_opt = sum(scores_optimized.values()) / len(scores_optimized) if scores_optimized else 0
    total_improvement = avg_opt - avg_orig
    
    print(f"{'-'*70}")
    print(f"{'综合得分':<20} {avg_orig:<15.3f} {avg_opt:<15.3f} {total_improvement:+.3f}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="N条HR问题RAGAS验证（优化版）")
    parser.add_argument("--num", "-n", type=int, default=5, help="测试问题数量")
    parser.add_argument("--all", "-a", action="store_true", help="测试全部问题")
    parser.add_argument("--compare", "-c", action="store_true", help="对比原版和优化版")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="输出目录（默认 ragas/results）",
    )
    args = parser.parse_args()

    print("="*70)
    print("        HR制度RAG - N条数据验证（优化版）")
    print("="*70)

    out_dir = args.output_dir or str(RESULTS_DIR)

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
    
    results_comparison = {}
    
    # 运行原版评估（如果需要对比）
    if args.compare:
        print("\n" + "="*70)
        print("第一阶段：原版RAG引擎评估")
        print("="*70)
        
        engine_original = RAGEngineOriginal(vectorstore)
        eval_data_original = run_rag_batch(questions, engine_original, "原版引擎")
        
        df_original = run_ragas_evaluation(eval_data_original)
        scores_original = calculate_average_scores(df_original)
        display_results(df_original, eval_data_original, "原版RAG - 评估结果")
        
        save_results(df_original, eval_data_original, out_dir, "original")
        results_comparison["original"] = scores_original
    
    # 运行优化版评估
    print("\n" + "="*70)
    if args.compare:
        print("第二阶段：优化版RAG引擎评估")
    else:
        print("运行优化版RAG引擎评估")
    print("="*70)
    
    engine_optimized = RAGEngineOptimized(vectorstore)
    eval_data_optimized = run_rag_batch(questions, engine_optimized, "优化版引擎")
    
    df_optimized = run_ragas_evaluation(eval_data_optimized)
    scores_optimized = calculate_average_scores(df_optimized)
    display_results(df_optimized, eval_data_optimized, "优化版RAG - 评估结果")
    
    save_results(df_optimized, eval_data_optimized, out_dir, "optimized")
    results_comparison["optimized"] = scores_optimized
    
    # 对比结果
    if args.compare and "original" in results_comparison:
        compare_versions(results_comparison["original"], results_comparison["optimized"])
    
    print("\n" + "="*70)
    print(f"[OK] 评估完成！结果已保存到 {out_dir}/ 目录")
    print("="*70)


if __name__ == "__main__":
    main()
