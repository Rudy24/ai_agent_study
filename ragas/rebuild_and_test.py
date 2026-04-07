# -*- coding: utf-8 -*-
"""
重建索引并做单条 RAG 验证（种子文档路径由 config / .env 决定）。
用法: 在项目根目录执行 python ragas/rebuild_and_test.py
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

_spec = importlib.util.spec_from_file_location(
    "_ragas_workspace_bootstrap",
    Path(__file__).resolve().parent / "bootstrap.py",
)
_boot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_boot)
PROJECT_ROOT = _boot.PROJECT_ROOT

os.chdir(PROJECT_ROOT)

from config import FAISS_INDEX_PATH, get_seed_document_path  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from rag_engine import RAGEngine  # noqa: E402


def main():
    """删除旧 FAISS 索引，按当前 knowledge_base 配置重建并测试一条问题。"""
    if os.path.exists(FAISS_INDEX_PATH):
        print("[INFO] 删除旧索引...")
        shutil.rmtree(FAISS_INDEX_PATH)
        print("[OK] 旧索引已删除")

    print("\n[INFO] 使用优化参数重建知识库...")
    kb = KnowledgeBase()

    doc_file = get_seed_document_path()
    if not os.path.exists(doc_file):
        print(f"[ERROR] 找不到文档: {doc_file}")
        sys.exit(1)

    texts = kb.process_document(doc_file)
    vectorstore = kb.create_vector_store(texts)

    print("\n" + "=" * 60)
    print("索引重建完成，运行单条测试...")
    print("=" * 60)

    test_case = {
        "question": "试用期一般为多久？",
        "ground_truth": "试用期为 1-3 个月。",
    }
    engine = RAGEngine(vectorstore)

    print(f"\n问题: {test_case['question']}")
    print(f"标准答案: {test_case['ground_truth']}")
    print("\n[INFO] 查询 RAG 系统...")

    result = engine.ask(test_case["question"])
    answer = result["result"]
    contexts = result["source_documents"]

    print(f"\nAI 回答:\n{answer}")
    print(f"\n参考来源数量: {len(contexts)}")
    for i, doc in enumerate(contexts, 1):
        print(f"\n[{i}] {doc.page_content[:150]}...")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
