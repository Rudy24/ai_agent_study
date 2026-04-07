# -*- coding: utf-8 -*-
"""
测试优化版引擎的单条问题效果（输出在控制台）。
用法: python ragas/test_single_optimized.py
"""
import importlib.util
import io
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
_boot.ensure_dirs()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import FAISS_INDEX_PATH  # noqa: E402
from knowledge_base import KnowledgeBase  # noqa: E402
from rag_engine_optimized import RAGEngineOptimized  # noqa: E402


def main():
    """加载索引并针对试用期问题跑一次优化版 RAG。"""
    print("=" * 60)
    print("优化版引擎单条测试")
    print("=" * 60)

    kb = KnowledgeBase()
    vectorstore = kb.load_vector_store()
    engine = RAGEngineOptimized(vectorstore)

    query = "试用期一般为多久？"
    print(f"\n问题: {query}")
    print("-" * 50)

    result = engine.ask(query)

    print(f"\nAI回答: {result['result']}")
    print(f"\n引用的文档数量: {len(result['source_documents'])}")

    print("\n引用的文档内容:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"\n[文档{i}]")
        print(doc.page_content[:200] + "...")
        print("-" * 50)


if __name__ == "__main__":
    main()
