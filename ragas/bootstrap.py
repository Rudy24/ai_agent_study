# -*- coding: utf-8 -*-
"""
RAGAS 评测目录路径配置（bootstrap.py）
====================================
本文件仅描述路径，不包含 pip 的 ragas 库；通过 importlib 加载，避免把 ragas/ 加入 sys.path 引发与第三方包同名冲突。
"""
from pathlib import Path

# 当前 ragas 评测根目录（本文件所在目录）
RAGAS_DIR = Path(__file__).resolve().parent
# 项目根目录（上一级，含 config.py、main.py）
PROJECT_ROOT = RAGAS_DIR.parent
# 评测用例 JSON 存放目录
DATA_DIR = RAGAS_DIR / "data"
# RAGAS 运行结果（Excel/JSON 等）默认输出目录
RESULTS_DIR = RAGAS_DIR / "results"
# 自动化流水线等中间产物目录
OUTPUT_DIR = RAGAS_DIR / "output"
# 默认 HR 评测问题集路径（当前 data/hr_eval_questions.json 为 50 问）
DEFAULT_QUESTIONS_JSON = DATA_DIR / "hr_eval_questions.json"
# 示例评估数据集（仅 RAGAS、不跑 RAG）
EXAMPLE_DATASET_JSON = DATA_DIR / "eval_dataset_example.json"


def ensure_dirs():
    """确保 data / results / output 目录存在。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_results_path(path_str: str) -> str:
    """
    若 path 为相对路径，则解析到 RESULTS_DIR 下，便于集中管理输出文件。
    """
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(RESULTS_DIR / p.name)
