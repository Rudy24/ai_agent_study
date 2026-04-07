# -*- coding: utf-8 -*-
"""兼容入口：实际脚本在 ragas/run_hr_eval.py。"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "ragas" / "run_hr_eval.py"),
        run_name="__main__",
    )
