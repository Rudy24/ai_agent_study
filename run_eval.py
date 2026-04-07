#!/usr/bin/env python
"""
统一评估入口脚本
用法：
    python run_eval.py              # 运行3条测试
    python run_eval.py 5            # 运行5条测试  
    python run_eval.py --all        # 运行全部测试
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

if __name__ == "__main__":
    # 默认运行 quick_eval
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("用法: python run_eval.py [数量] [--all]")
        print("  python run_eval.py          # 默认3条")
        print("  python run_eval.py 5        # 5条")
        print("  python run_eval.py --all    # 全部")
        sys.exit(0)
    
    # 运行 ragas/quick_eval.py
    quick_eval_path = root / "ragas" / "quick_eval.py"
    if quick_eval_path.exists():
        import runpy
        sys.argv = [str(quick_eval_path)] + sys.argv[1:]
        runpy.run_path(str(quick_eval_path), run_name="__main__")
    else:
        print(f"错误: 找不到 {quick_eval_path}")
        sys.exit(1)
