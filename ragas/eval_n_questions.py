"""
N条HR问题RAGAS验证脚本（支持传参）
================================
用法:
  python ragas/eval_n_questions.py --num 10
  python ragas/eval_n_questions.py --num 5 --output results.xlsx
  python ragas/eval_n_questions.py --all

参数:
  --num: 指定测试问题数量（默认5条）
  --all: 测试全部问题
  --input: 输入问题集（默认 ragas/data/hr_eval_questions.json）
  --output: 输出 Excel 文件名（默认写入 ragas/results/）
  --save-dataset: 保存评估数据集到 JSON（同目录）
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# 项目根目录加入 path，便于导入 config（须在 import ragas 库之前）
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# 加载本目录 bootstrap（不用 import ragas.bootstrap，避免与 pip 的 ragas 冲突）
_spec = importlib.util.spec_from_file_location(
    "_ragas_workspace_bootstrap",
    Path(__file__).resolve().parent / "bootstrap.py",
)
_boot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_boot)
DATA_DIR = _boot.DATA_DIR
RESULTS_DIR = _boot.RESULTS_DIR
DEFAULT_QUESTIONS_JSON = _boot.DEFAULT_QUESTIONS_JSON
resolve_results_path = _boot.resolve_results_path
_boot.ensure_dirs()

import argparse
import io

# 设置编码（Windows 控制台 UTF-8）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 导入项目模块
from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    LOCAL_MODEL_PATH,
    RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN,
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


def load_questions(input_file: str, num: int = None) -> List[Dict]:
    """加载测试问题"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions = data["questions"]
    
    if num and num > 0:
        # 如果指定数量大于总数，则取全部
        actual_num = min(num, len(questions))
        questions = questions[:actual_num]
    
    print(f"[INFO] 从 {input_file} 加载了 {len(questions)} 个测试问题")
    if num and num > len(questions):
        print(f"[WARN] 请求数量 {num} 大于实际数量，已取全部 {len(questions)} 条")
    
    return questions


def run_rag_on_questions(questions: List[Dict], engine: RAGEngine) -> List[Dict]:
    """运行RAG获取答案"""
    results = []
    total = len(questions)
    
    print(f"\n[INFO] 开始运行RAG（共 {total} 个问题）...")
    print("-" * 70)

    for i, item in enumerate(questions, 1):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"  [{i}/{total}] {question[:35]}...", end=" ", flush=True)
        try:
            result = engine.ask(question)
            # 有推理前缀时仅用最终答案段做 RAGAS，避免指标被推理文本拉偏
            answer = result.get("answer_only") or result.get("result", "")
            contexts = [d.page_content for d in result.get("source_documents", [])]

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
            })
            print("OK")
        except Exception as e:
            print(f"FAIL: {str(e)[:50]}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": "",
                "contexts": [],
            })

    print("-" * 70)
    success_count = len([r for r in results if r["answer"]])
    print(f"[OK] 成功处理 {success_count}/{total} 条")
    return results


def run_ragas_evaluation(eval_data: List[Dict]) -> pd.DataFrame:
    """运行RAGAS评估"""
    # 准备数据集
    for item in eval_data:
        item.setdefault("contexts", [])
        item.setdefault("ground_truth", item.get("answer", ""))

    dataset = Dataset.from_list(eval_data)

    # 设置评估器（answer_relevancy strictness / 生成侧扩窗见 .env：RAGAS_ANSWER_RELEVANCY_STRICTNESS、RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN）
    _rs = os.getenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1").strip() or "1"
    print(
        f"\n[INFO] 初始化RAGAS评估器（DeepSeek n=1；answer_relevancy strictness={_rs}；RAG 扩窗 margin={RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN}）..."
    )
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
    print("[WARNING] 需要调用 DeepSeek API，请耐心等待...")
    print("-" * 70)

    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return result.to_pandas()


def show_results(df: pd.DataFrame, eval_data: List[Dict], output_file: str):
    """显示评估结果并保存"""
    # 获取分数列
    score_cols = []
    for col in df.columns:
        if col not in ['question', 'answer', 'contexts', 'ground_truth', 'user_input', 
                       'retrieved_contexts', 'reference', 'response']:
            score_cols.append(col)

    print("\n" + "=" * 70)
    print(f"              RAGAS 评估结果 - {len(eval_data)} 条HR问题")
    print("=" * 70)

    # 显示每个问题的详细结果
    for i, row in df.iterrows():
        print(f"\n【问题 {i+1}】{eval_data[i]['question']}")
        print(f"标准答案: {eval_data[i]['ground_truth'][:50]}...")
        print(f"AI回答: {eval_data[i]['answer'][:70]}...")
        print("-" * 50)
        
        for col in score_cols:
            score = row[col]
            # 处理可能的数组类型
            if hasattr(score, '__len__') and not isinstance(score, str):
                score = score[0] if len(score) > 0 else None
            
            if pd.notna(score) and isinstance(score, (int, float)):
                filled = int(float(score) * 20)
                bar = "#" * filled + "-" * (20 - filled)
                print(f"  {col:20s}: {score:.3f} [{bar}]")
            else:
                print(f"  {col:20s}: N/A")

    # 显示平均分
    print("\n" + "=" * 70)
    print("                    平均分汇总")
    print("=" * 70)
    
    total_score = 0
    valid_count = 0
    
    for col in score_cols:
        try:
            # 转换为数值并计算平均
            values = []
            for val in df[col]:
                if hasattr(val, '__len__') and not isinstance(val, str):
                    val = val[0] if len(val) > 0 else None
                if pd.notna(val) and isinstance(val, (int, float)):
                    values.append(float(val))
            
            if values:
                avg = sum(values) / len(values)
                filled = int(avg * 20)
                bar = "#" * filled + "-" * (20 - filled)
                print(f"  {col:20s}: {avg:.3f} [{bar}]")
                total_score += avg
                valid_count += 1
            else:
                print(f"  {col:20s}: N/A")
        except Exception as e:
            print(f"  {col:20s}: Error")
    
    # 主指标：faithfulness 与 answer_relevancy 两列各自均值再取平均；四项均值为辅
    if valid_count > 0:
        overall = total_score / valid_count

        def _column_mean(name: str):
            if name not in df.columns:
                return None
            vals = []
            for val in df[name]:
                if hasattr(val, "__len__") and not isinstance(val, str):
                    val = val[0] if len(val) > 0 else None
                if pd.notna(val) and isinstance(val, (int, float)):
                    vals.append(float(val))
            return sum(vals) / len(vals) if vals else None

        fm = _column_mean("faithfulness")
        am = _column_mean("answer_relevancy")
        print("-" * 70)
        if fm is not None and am is not None:
            pavg = (fm + am) / 2
            print(f"  {'主指标(忠实+相关)':20s}: {pavg:.3f}  （faithfulness 第一，answer_relevancy 第二；分项 faithfulness={fm:.3f}, answer_relevancy={am:.3f}）")
        elif fm is not None or am is not None:
            pavg = (fm if fm is not None else 0) + (am if am is not None else 0)
            div = int(fm is not None) + int(am is not None)
            print(f"  {'主指标(忠实+相关)':20s}: {pavg / div:.3f}")
        print(f"  {'四项均值(参考)':20s}: {overall:.3f}")

    # 保存结果
    print("\n" + "=" * 70)
    print("                     保存结果")
    print("=" * 70)
    
    try:
        # 保存Excel
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"[OK] Excel结果已保存: {output_file}")
    except Exception as e:
        print(f"[WARN] 保存Excel失败: {e}")
    
    try:
        # 保存JSON
        json_file = output_file.replace('.xlsx', '.json')
        df.to_json(json_file, orient="records", force_ascii=False, indent=2)
        print(f"[OK] JSON结果已保存: {json_file}")
    except Exception as e:
        print(f"[WARN] 保存JSON失败: {e}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="N条HR问题RAGAS验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ragas/eval_n_questions.py --num 10
  python ragas/eval_n_questions.py --all
  python ragas/eval_n_questions.py --num 5 --output my_results.xlsx
  python ragas/eval_n_questions.py --num 3 --save-dataset
        """
    )
    
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=5,
        help="测试问题数量（默认5条）"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="测试全部问题（覆盖--num参数）"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="输入问题集 JSON（默认 ragas/data/hr_eval_questions.json）",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="eval_results.xlsx",
        help="输出 Excel 文件名（相对路径写入 ragas/results/）",
    )
    
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="保存评估数据集到JSON文件"
    )
    
    args = parser.parse_args()

    print("=" * 70)
    print("        HR制度RAG - N条数据验证脚本")
    print("=" * 70)

    # 默认用例路径：ragas/data/hr_eval_questions.json；相对名在 ragas/data 下再解析一次
    input_file = args.input or str(DEFAULT_QUESTIONS_JSON)
    if not os.path.isfile(input_file) and args.input:
        alt = DATA_DIR / Path(args.input).name
        if alt.is_file():
            input_file = str(alt)
    if not os.path.isfile(input_file):
        print(f"[ERROR] 找不到输入文件: {input_file}")
        sys.exit(1)

    output_file = resolve_results_path(args.output)
    
    # 确定测试数量
    test_num = None if args.all else args.num
    
    # 加载问题
    questions = load_questions(input_file, test_num)
    
    if not questions:
        print("[ERROR] 没有加载到任何问题")
        sys.exit(1)
    
    # 显示问题列表
    print("\n测试问题列表:")
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
        print("[HINT] 请先运行: python main.py 构建索引")
        sys.exit(1)

    # 创建RAG引擎
    engine = RAGEngine(vectorstore)

    # 运行RAG
    eval_data = run_rag_on_questions(questions, engine)

    # 保存数据集（如果指定）
    if args.save_dataset:
        dataset_file = str(
            RESULTS_DIR / (Path(output_file).stem + "_dataset.json")
        )
        with open(dataset_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] 评估数据集已保存: {dataset_file}")

    # 运行RAGAS评估
    df = run_ragas_evaluation(eval_data)

    # 显示结果
    show_results(df, eval_data, output_file)

    print("\n[OK] 评估完成!")


if __name__ == "__main__":
    main()
