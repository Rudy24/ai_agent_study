"""
重建FAISS索引优化版
==================
1. 增大chunk_size确保试用期关键信息完整保留
2. 保留完整段落，避免切断关键内容
3. 添加试用期相关内容的前缀标记
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, LOCAL_MODEL_PATH, get_seed_document_path
from knowledge_base import KnowledgeBase

def main():
    print("="*70)
    print("重建FAISS索引（优化版）")
    print("="*70)
    
    # 删除旧索引
    if os.path.exists(FAISS_INDEX_PATH):
        import shutil
        shutil.rmtree(FAISS_INDEX_PATH)
        print(f"[OK] 已删除旧索引: {FAISS_INDEX_PATH}")
    
    # 创建知识库
    kb = KnowledgeBase()
    
    # 处理文档（路径由 config / .env 的 DOCS_DIR、SEED_DOCUMENT_PATH 等决定）
    file_path = get_seed_document_path()
    if not os.path.exists(file_path):
        print(f"[ERROR] 找不到文档: {file_path}")
        sys.exit(1)

    print(f"[START] 处理文档: {file_path}")
    
    # 使用知识库的process_document处理
    texts = kb.process_document(file_path)
    
    print(f"\n[SUMMARY] 文档已切分为 {len(texts)} 个片段")
    print("\n预览前10个片段:")
    for i, text in enumerate(texts[:10], 1):
        content = text.page_content[:80].replace('\n', ' ')
        has_probation = "试用" in text.page_content
        marker = " [★含试用期内容]" if has_probation else ""
        print(f"  {i}. {content}...{marker}")
    
    # 查找包含试用期的片段
    probation_chunks = [t for t in texts if "试用" in t.page_content]
    print(f"\n[INFO] 共有 {len(probation_chunks)} 个片段包含'试用期'相关内容")
    
    if probation_chunks:
        print("\n试用期相关片段内容:")
        for i, chunk in enumerate(probation_chunks[:5], 1):
            print(f"\n[片段{i}]")
            print(chunk.page_content)
            print("-"*50)
    
    # 创建向量索引
    print("\n[INFO] 创建向量索引...")
    vectorstore = kb.create_vector_store(texts)
    
    print(f"\n[OK] 索引重建完成！")
    print(f"[OK] 索引路径: {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()
