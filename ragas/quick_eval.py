# -*- coding: utf-8 -*-
"""
快捷评测入口：把简短参数转成 eval_n_questions.py 所需 argv 并执行（避免自引用 run_path）。
用法:
  python ragas/quick_eval.py           # 默认 3 条
  python ragas/quick_eval.py 10        # 10 条
  python ragas/quick_eval.py --all       # 全部
项目根: python run_eval.py [同上]
"""
import argparse  # 解析命令行
import runpy  # 以 __main__ 方式执行目标脚本
import sys  # 改写 argv 供子脚本使用
from pathlib import Path  # 拼 eval_n_questions 路径

# 项目根目录（ragas 上一级），便于导入 config 等
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> None:
    """解析参数后转发到 eval_n_questions.py。"""
    parser = argparse.ArgumentParser(
        description="快捷运行 RAGAS 评测（内部调用 eval_n_questions.py）",
    )
    # 可选位置参数：条数；不传则与 run_eval 约定一致为 3
    parser.add_argument(
        "count",
        nargs="?",
        type=int,
        default=None,
        metavar="N",
        help="测试问题条数；省略则默认 3",
    )
    parser.add_argument("--all", "-a", action="store_true", help="评测全部问题")
    parser.add_argument("--input", "-i", type=str, default=None, help="问题集 JSON 路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="结果 Excel 文件名")
    parser.add_argument("--save-dataset", action="store_true", help="保存评估数据集 JSON")

    args = parser.parse_args()

    # 目标脚本必须与 quick_eval 同目录，禁止再指向 quick_eval 自身
    eval_script = Path(__file__).resolve().parent / "eval_n_questions.py"
    if not eval_script.is_file():
        print(f"[ERROR] 找不到脚本: {eval_script}")
        sys.exit(1)

    # 组装 eval_n_questions 的 sys.argv[0] 为脚本路径，其后为选项
    argv = [str(eval_script)]
    if args.all:
        argv.append("--all")
    else:
        n = args.count if args.count is not None else 3
        argv.extend(["--num", str(n)])
    if args.input:
        argv.extend(["--input", args.input])
    if args.output:
        argv.extend(["--output", args.output])
    if args.save_dataset:
        argv.append("--save-dataset")

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        runpy.run_path(str(eval_script), run_name="__main__")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
