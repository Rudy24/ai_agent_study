"""
RAGAS 评估模块（eval_ragas.py）
=============================
作用：使用 RAGAS 框架定量评估 RAG 系统性能。

评估指标：
  - 上下文精确度 (context_precision): 检索到的文档中有多少与答案相关
  - 上下文召回率 (context_recall): 答案中信息有多少来自检索到的文档
  - 上下文相关性 (context_relevancy): 检索到的文档与问题的相关程度
  - 答案忠实度 (faithfulness): 答案是否基于检索到的文档，是否胡编
  - 答案相关性 (answer_relevancy): 答案是否直接回答问题了
  - 答案正确性 (answer_correctness): 答案与标准答案的匹配程度

用法：
  1. 准备评估数据集（JSON格式，包含 question, answer, contexts, ground_truth）
  2. 运行：python ragas/eval_ragas.py --input ragas/data/eval_dataset_example.json
  3. 查看生成的 Excel 报告

示例数据集格式见 ragas/data/eval_dataset_example.json
"""
import importlib.util
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 项目根加入 path
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
DATA_DIR = _boot.DATA_DIR
EXAMPLE_DATASET_JSON = _boot.EXAMPLE_DATASET_JSON
resolve_results_path = _boot.resolve_results_path
_boot.ensure_dirs()

import pandas as pd
from datasets import Dataset

# 导入 RAGAS 评估指标
try:
    from ragas import evaluate
    # RAGAS 0.4.x 指标导入
    try:
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
        )
        # answer_correctness 在 0.4.x 中不可用
        try:
            from ragas.metrics import answer_correctness
        except ImportError:
            answer_correctness = None
    except ImportError as e:
        print(f"[ERROR] Failed to import RAGAS metrics: {e}")
        sys.exit(1)
    from ragas.llms import LangchainLLMWrapper
except ImportError as e:
    print(f"[ERROR] RAGAS import failed: {e}")
    print("[HINT] Run: pip install ragas")
    sys.exit(1)

from config import (
    EMBEDDING_MODEL_NAME,
    LOCAL_MODEL_PATH,
    get_ragas_answer_relevancy,
    get_ragas_evaluator_llm,
)


def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    """
    加载评估数据集。
    格式：[{"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}, ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_ragas_dataset(eval_data: List[Dict[str, Any]]) -> Dataset:
    """
    将评估数据转换为 RAGAS 所需的 Dataset 格式。
    RAGAS 期望的列：question, answer, contexts, ground_truth
    """
    # 确保每个样本都有 contexts 字段（如果没有，设为空列表）
    for item in eval_data:
        if "contexts" not in item:
            item["contexts"] = []
        if "ground_truth" not in item:
            item["ground_truth"] = item.get("answer", "")  # 如果没有标准答案，用答案代替

    # 创建 HuggingFace Dataset
    dataset = Dataset.from_list(eval_data)
    return dataset


def setup_evaluator():
    """
    配置 RAGAS 使用的 LLM 和 Embedding 模型。
    RAGAS 需要 LLM 来判断答案质量，以及 Embedding 来计算语义相似度。
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = get_ragas_evaluator_llm()

    # 优先使用本地模型路径，避免连接 HuggingFace Hub
    import os
    model_path = LOCAL_MODEL_PATH if os.path.isdir(LOCAL_MODEL_PATH) else EMBEDDING_MODEL_NAME
    if model_path == LOCAL_MODEL_PATH:
        print(f"[INFO] 评估器使用本地嵌入模型: {model_path}")
    else:
        print(f"[WARN] 本地模型未找到，将尝试从 HuggingFace 下载: {EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 包装为 RAGAS 可用的格式
    from ragas.embeddings import LangchainEmbeddingsWrapper

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return evaluator_llm, evaluator_embeddings


def run_evaluation(dataset: Dataset, evaluator_llm, evaluator_embeddings) -> pd.DataFrame:
    """
    执行 RAGAS 评估，返回结果 DataFrame。
    """
    # 定义要使用的评估指标（RAGAS 0.4.x 支持的指标）
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        get_ragas_answer_relevancy(),
    ]
    # 注：answer_correctness 和 context_relevancy 在 RAGAS 0.4.x 中不可用

    print("[INFO] 开始 RAGAS 评估...")
    print(f"[INFO] 评估样本数: {len(dataset)}")
    print(f"[INFO] 评估指标: {[m.name for m in metrics]}")

    # 执行评估
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 转换为 DataFrame
    df = result.to_pandas()

    # 添加平均分数行
    numeric_cols = [col for col in df.columns if col not in ["question", "answer", "contexts", "ground_truth"]]
    avg_row = {col: df[col].mean() for col in numeric_cols}
    avg_row["question"] = "[AVERAGE]"
    avg_row["answer"] = ""
    avg_row["contexts"] = []
    avg_row["ground_truth"] = ""

    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    return df


def save_results(df: pd.DataFrame, output_path: str):
    """
    保存评估结果到 Excel 文件。
    """
    # 处理 contexts 列（列表转字符串，便于 Excel 显示）
    df_display = df.copy()
    if "contexts" in df_display.columns:
        df_display["contexts"] = df_display["contexts"].apply(
            lambda x: "\n---\n".join(x) if isinstance(x, list) and x else ""
        )

    # 保存到 Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_display.to_excel(writer, sheet_name="RAGAS Results", index=False)

        # 创建汇总统计表
        numeric_cols = [col for col in df.columns if col not in ["question", "answer", "contexts", "ground_truth"]]
        summary = df[numeric_cols].describe()
        summary.to_excel(writer, sheet_name="Statistics")

    print(f"[SUCCESS] 评估结果已保存: {output_path}")


def print_summary(df: pd.DataFrame):
    """
    在控制台打印评估结果摘要。
    """
    numeric_cols = [col for col in df.columns if col not in ["question", "answer", "contexts", "ground_truth"]]

    print("\n" + "=" * 60)
    print("RAGAS 评估结果摘要")
    print("=" * 60)

    for col in numeric_cols:
        avg = df[col].mean()
        print(f"  {col:30s}: {avg:.3f}")

    print("=" * 60)
    print("评分范围: 0.0 (最差) ~ 1.0 (最佳)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAGAS RAG Evaluation System")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="评估数据集 JSON（默认 ragas/data/eval_dataset_example.json）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="eval_results.xlsx",
        help="输出 Excel（相对路径写入 ragas/results/）",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="是否在控制台显示详细结果",
    )

    args = parser.parse_args()

    in_path = args.input or str(EXAMPLE_DATASET_JSON)
    if not os.path.isfile(in_path) and args.input:
        alt = DATA_DIR / Path(args.input).name
        if alt.is_file():
            in_path = str(alt)
    if not os.path.isfile(in_path):
        print(f"[ERROR] 评估数据集不存在: {in_path}")
        print("[HINT] 请参考 ragas/data/eval_dataset_example.json")
        sys.exit(1)

    out_path = resolve_results_path(args.output)

    # 加载数据
    print(f"[INFO] 加载评估数据集: {in_path}")
    eval_data = load_eval_dataset(in_path)
    print(f"[INFO] 加载了 {len(eval_data)} 条评估样本")

    # 创建 RAGAS Dataset
    dataset = create_ragas_dataset(eval_data)

    # 设置评估器
    print("[INFO] 初始化评估模型...")
    try:
        evaluator_llm, evaluator_embeddings = setup_evaluator()
    except Exception as e:
        print(f"[ERROR] 初始化评估模型失败: {e}")
        print("[HINT] 请检查 DeepSeek API 配置和本地嵌入模型")
        sys.exit(1)

    # 执行评估
    try:
        result_df = run_evaluation(dataset, evaluator_llm, evaluator_embeddings)
    except Exception as e:
        print(f"[ERROR] 评估过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 保存结果
    save_results(result_df, out_path)

    # 打印摘要
    print_summary(result_df)

    if args.show:
        print("\n详细结果:")
        print(result_df.to_string())


if __name__ == "__main__":
    main()
