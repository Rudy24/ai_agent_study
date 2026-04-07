"""
修复 Windows 编码问题的工具脚本
===============================
用法: python fix_encoding.py

功能:
1. 自动修复所有项目文件中的 Unicode 字符（Emoji、特殊符号等）
2. 替换为 ASCII 安全字符
"""
import os
import re

# 定义替换映射
REPLACEMENTS = {
    '🔨': '[Build]',
    '✓': '[OK]',
    '✅': '[OK]',
    '🚀': '[Start]',
    '🤖': '[AI]',
    '👤': '[User]',
    '📚': '[Doc]',
    '█': '#',  # 进度条方块替换为#
    '░': '-',  # 进度条空白替换为-
    '': '?',  # 未知字符
}

def fix_file(filepath):
    """修复单个文件的编码问题"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # 替换所有 Unicode 字符
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[FIXED] {filepath}")
        return True
    return False

def main():
    """修复所有 Python 文件"""
    files_to_fix = [
        'main.py',
        'knowledge_base.py',
        'rag_engine.py',
        'eval_5_questions.py',
        'test_single.py',
        'eval_hr_30.py',
        'auto_eval_pipeline.py',
        'run_hr_eval.py',
        'generate_eval_dataset.py',
    ]

    print("=" * 60)
    print("修复 Windows 编码问题")
    print("=" * 60)

    fixed_count = 0
    for filename in files_to_fix:
        if os.path.exists(filename):
            if fix_file(filename):
                fixed_count += 1
        else:
            print(f"[SKIP] 文件不存在: {filename}")

    print("=" * 60)
    print(f"修复完成！共修复 {fixed_count} 个文件")
    print("=" * 60)
    print("\n现在可以直接运行脚本，不会再出现编码错误")

if __name__ == "__main__":
    main()
