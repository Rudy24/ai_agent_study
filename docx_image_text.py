# -*- coding: utf-8 -*-
"""
Word（.docx）正文插图 → 可检索文本
================================
按文档顺序遍历段落与表格单元格，将 run 内嵌图片导出为字节后做 OCR，
在原文流中插入「【插图文字识别】…」段落，供 knowledge_base 切块与嵌入。

依赖（按需安装）：
- python-docx：解析 OOXML
- Pillow：解码 png/jpeg 等
- rapidocr-onnxruntime：中文 OCR 较稳，纯 pip（体积较大）
- 或 pytesseract + 本机安装 Tesseract，并配置 lang=chi_sim
"""
from __future__ import annotations

import io  # 内存中打开图片二进制
from typing import List, Optional, Union  # 类型

from docx import Document  # 打开 docx
from docx.document import Document as DocumentObject  # 类型判断
from docx.oxml.ns import qn  # r:embed 等 QName
from docx.table import Table, _Cell  # 表格与单元格
from docx.text.paragraph import Paragraph  # 段落


def _iter_block_items(parent: Union[DocumentObject, _Cell]):
    """按 Word 文档顺序产出段落与表格（含表格内嵌套）。"""
    if isinstance(parent, DocumentObject):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        return
    for child in parent_elm.iterchildren():
        tag = child.tag
        if tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif tag.endswith("}tbl"):
            tbl = Table(child, parent)
            for row in tbl.rows:
                for cell in row.cells:
                    yield from _iter_block_items(cell)


def _image_blobs_from_run(run, document_part) -> List[bytes]:
    """从一个 run 的 w:drawing / VML 中解析出所有内嵌图片的二进制。"""
    blobs: List[bytes] = []
    el = run._element
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    for blip in el.findall(".//a:blip", ns):
        rid = blip.get(qn("r:embed"))
        if not rid or rid not in document_part.rels:
            continue
        rel = document_part.rels[rid]
        rt = rel.reltype or ""
        if "image" not in rt and "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" not in rt:
            continue
        try:
            blobs.append(rel.target_part.blob)
        except Exception:
            continue
    # 旧式 VML：imagedata（命名空间因版本而异，用后缀匹配）
    for node in el.iter():
        tag = node.tag
        if not tag.endswith("imagedata"):
            continue
        rid = node.get(qn("r:id")) or node.get(qn("r:embed"))
        if not rid or rid not in document_part.rels:
            continue
        rel = document_part.rels[rid]
        try:
            blobs.append(rel.target_part.blob)
        except Exception:
            continue
    return blobs


def _ocr_image_bytes(blob: bytes) -> str:
    """对单张图片做 OCR；无可用引擎时返回空串。"""
    if not blob:
        return ""
    # 1) RapidOCR（推荐中文场景）
    try:
        import numpy as np  # RapidOCR 常用 ndarray 输入
        from PIL import Image  # 解码位图
        from rapidocr_onnxruntime import RapidOCR  # ONNX 推理

        img = Image.open(io.BytesIO(blob)).convert("RGB")
        arr = np.array(img)
        ocr = RapidOCR()
        res, _ = ocr(arr)
        if not res:
            return ""
        lines: List[str] = []
        for item in res:
            if len(item) >= 2 and item[1]:
                lines.append(str(item[1]).strip())
        return "\n".join(x for x in lines if x)
    except ImportError:
        pass
    except Exception as ex:
        print(f"[WARN] RapidOCR 识别单图失败: {ex}")
    # 2) Tesseract 兜底
    try:
        import pytesseract  # 需本机安装 tesseract
        from PIL import Image

        img = Image.open(io.BytesIO(blob))
        return (pytesseract.image_to_string(img, lang="chi_sim+eng") or "").strip()
    except ImportError:
        pass
    except Exception as ex:
        print(f"[WARN] pytesseract 识别单图失败: {ex}")
    return ""


def docx_to_plain_text_with_image_ocr(file_path: str) -> str:
    """
    读取 .docx：正文按顺序拼接段落文字，并在每张插图位置插入 OCR 文本块。
    返回一整段字符串，供与现有 RecursiveCharacterTextSplitter 衔接。
    """
    doc = Document(file_path)
    part = doc.part
    out: List[str] = []
    for block in _iter_block_items(doc):
        if not isinstance(block, Paragraph):
            continue
        para_chunks: List[str] = []
        for run in block.runs:
            t = run.text or ""
            if t:
                para_chunks.append(t)
            for blob in _image_blobs_from_run(run, part):
                ocr_txt = _ocr_image_bytes(blob)
                if ocr_txt:
                    para_chunks.append(f"\n【插图文字识别】\n{ocr_txt}\n")
                else:
                    para_chunks.append("\n【插图】（当前环境未识别到文字，可安装 rapidocr-onnxruntime 或 Tesseract）\n")
        merged = "".join(para_chunks).strip()
        if merged:
            out.append(merged)
    return "\n\n".join(out)
