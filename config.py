"""
全局配置模块（config.py）
------------------------
作用：集中管理嵌入模型名、DeepSeek 对话参数、检索条数、FAISS 索引路径。
数据流：启动时 load_dotenv() 读取 .env；各模块 `from config import XXX` 即可用常量。

【阅读分区 — 建议自上而下】
  A) 线程与环境：KMP/OMP/MKL 等，须在首次 import torch 前生效（防 Windows 多库冲突段错误）。
  B) 嵌入与 DeepSeek：`EMBEDDING_*`、`LOCAL_MODEL_PATH`、`DEEPSEEK_*`。
  C) CrossEncoder：`RERANKER_MODEL` → 默认本地目录 `RERANKER_MODEL_PATH`。
  D) 工具：`_env_bool`、`_env_int`（读 .env 整数/布尔并钳位）。
  E) 检索与延迟：`SEARCH_K`、`RERANK_*`、`USE_RERANKER`、`RAG_LOW_LATENCY_MODE`、`RAG_LLM_MAX_TOKENS`。
  F) 行为开关：`RAG_PROMPT_STYLE`、`RAG_REASONING_MODE`（COT/ReAct）、离题短路 `RAG_OFF_TOPIC_*`、`DOCS_PUBLIC_BASE_URL`。
  G) RAGAS：评测用 LLM、answer_relevancy 等工厂函数（脚本用，非 API 热路径）。
  H) 运行与模板：`API_*`、`FAISS_INDEX_PATH`、种子文档解析、`get_current_prompt_template()`。

与架构说明对应：docs/ARCHITECTURE.md 第 4、7 节及「配置入口」表。
"""
import os  # 读取环境变量
from dotenv import load_dotenv  # 加载 .env 到 os.environ

# 将项目根目录下的 .env 合并进环境（若不存在则静默跳过）
load_dotenv()

# ---------- A. 线程与安全默认值（必须在首次 import torch / 含 native 的库之前）----------
# Windows / 多原生库：PyTorch、NumPy、FAISS、rank-bm25 各带 OpenMP 时易冲突导致段错误；须在首次 import torch / langchain 之前生效
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 强制 NumExpr 也单线程，减少 Windows 多进程/多 import 下的段错误
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------- B. 嵌入模型与本地缓存路径 ----------
# --- 嵌入模型（本地 HuggingFace，无需为向量付 API 费）---
# 可选模型（按性能排序）：
# 1. BAAI/bge-large-zh-v1.5 - 性能最好，需要更多内存
# 2. BAAI/bge-base-zh-v1.5 - 平衡性能和资源
# 3. BAAI/bge-small-zh-v1.5 - 轻量级，资源占用小
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 升级为大模型，提高检索准确性
# 嵌入推理设备：auto / cuda / cpu（.env 设 cuda 可强制用 GPU；不可用时 knowledge_base 会告警并回退 cpu）
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
# HuggingFace 国内镜像 endpoint（解决 huggingface.co 连接超时），如 https://hf-mirror.com
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "").strip().rstrip("/")
# 本地模型缓存目录（若已手动下载模型，可直接指向本地路径，跳过网络下载）
# 默认指向项目目录下的 models/bge-small-zh-v1.5，方便统一管理
_DEFAULT_LOCAL_MODEL = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", _DEFAULT_LOCAL_MODEL).strip()

# ---------- B2. DeepSeek（主 RAG 与部分评测共用 base_url / model 名）----------
# --- DeepSeek 对话 LLM（OpenAI 兼容接口）---
# API 密钥来自 .env 的 DEEPSEEK_API_KEY，勿写死在代码里
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 模型名按 DeepSeek 开放平台文档填写
DEEPSEEK_MODEL = "deepseek-chat"
# 基础 URL 需带 /v1，与 LangChain OpenAI 封装的路径约定一致
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ---------- C. CrossEncoder 本地权重路径（rag_engine 启动时加载）----------
# --- CrossEncoder 重排（bge-reranker-large 等，sentence-transformers）---
# RERANKER_MODEL：环境变量 RERANKER_MODEL 可选 base / large，未设时默认 base
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "base").strip().lower()

if RERANKER_MODEL == "large":
    _DEFAULT_RERANKER = os.path.join(os.path.dirname(__file__), "models", "bge-reranker-large")
    print(f"[INFO] 使用 bge-reranker-large 模型")
else:
    _DEFAULT_RERANKER = os.path.join(os.path.dirname(__file__), "models", "bge-reranker-base")
    print(f"[INFO] 使用 bge-reranker-base 模型（推荐，速度更快）")

RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", _DEFAULT_RERANKER).strip()


# ---------- D. 环境变量解析工具（供下方整数/布尔配置复用）----------
def _env_bool(name: str, default: bool) -> bool:
    """解析环境变量布尔值：空串用 default，0/false/no/off 为 False。"""
    v = os.getenv(name, "").strip().lower()
    if v == "":
        return default
    return v not in ("0", "false", "no", "off", "n")


def _env_int(name: str, default: int, min_v: int = 0, max_v: int | None = None) -> int:
    """读取整数环境变量并钳位。"""
    try:
        v = int(os.getenv(name, str(default)).strip())
    except ValueError:
        v = default
    v = max(min_v, v)
    if max_v is not None:
        v = min(max_v, v)
    return v


# ---------- E. 检索与重排数量、低开销模式、LLM 输出上限（rag_engine 热路径）----------
# --- 检索与重排序相关数量（依赖 _env_int，故放在解析函数之后）---
# SEARCH_K：向量检索与 BM25 各自召回的文档条数（可由 .env 的 SEARCH_K 覆盖）
SEARCH_K = _env_int("SEARCH_K", 25, 4, 80)
# RERANK_TOP_N：最终进入 LLM 的文档条数（可由 .env 覆盖）
RERANK_TOP_N = _env_int("RERANK_TOP_N", 5, 1, 12)
# 以下两项为预留阈值：当前 rag_engine 检索路径未读取（过滤主要靠 SEARCH_K 与重排 Top-N）
SIMILARITY_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.5

USE_RERANKER = _env_bool("USE_RERANKER", True)
# 重排前向量与 BM25 各取的条数上限（合并去重后参与 CrossEncoder；越大越稳、越慢）
RERANK_POOL_SIZE = _env_int("RERANK_POOL_SIZE", 16, 4, 80)
# CrossEncoder.predict 批次大小（GPU 可试 16~32；CPU 或显存不足调小）
RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 8, 1, 64)
# 送入 CrossEncoder 的单条 passage 最大字符（与 ~512 token 对齐，显著降 tokenizer/推理耗时）；0 表示不截断
RERANK_MAX_PASSAGE_CHARS = _env_int("RERANK_MAX_PASSAGE_CHARS", 1200, 0, 32000)
# CUDA 上 CrossEncoder 是否用 FP16（通常更快更省显存；异常时可设 false）
RERANK_USE_FP16_CUDA = _env_bool("RERANK_USE_FP16_CUDA", True)
# CrossEncoder 设备：auto / cuda / cpu（与 EMBEDDING_DEVICE 独立；.env 设 cuda 强制 GPU，不可用时 rag_engine 回退 cpu）
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto").strip().lower()

# 低开销模式：高并发或压测时收紧检索/重排与 LLM 输出上限，降低端到端延迟（精度可能略降）
RAG_LOW_LATENCY_MODE = _env_bool("RAG_LOW_LATENCY_MODE", False)
if RAG_LOW_LATENCY_MODE:
    # 限制 CrossEncoder 候选条数，减少 predict 前向次数
    RERANK_POOL_SIZE = min(RERANK_POOL_SIZE, 10)
    # 缩短单条 passage，降低 tokenizer 与推理耗时；原为 0（不截断）时改为有界截断
    if RERANK_MAX_PASSAGE_CHARS > 0:
        RERANK_MAX_PASSAGE_CHARS = min(RERANK_MAX_PASSAGE_CHARS, 800)
    else:
        RERANK_MAX_PASSAGE_CHARS = 800
    # 收窄双路检索与进 Prompt 的文档数
    SEARCH_K = min(SEARCH_K, 15)
    RERANK_TOP_N = min(RERANK_TOP_N, 4)

# 主 RAG 链路 LLM 输出 token 上限（越小通常越快；过小易截断长答）
RAG_LLM_MAX_TOKENS = _env_int("RAG_LLM_MAX_TOKENS", 1024, 128, 8192)
if RAG_LOW_LATENCY_MODE:
    # 与 concise 场景对齐，避免生成过长段落拖高延迟
    RAG_LLM_MAX_TOKENS = min(RAG_LLM_MAX_TOKENS, 640)

# RAG 评测与调优目标：faithfulness 第一（不可牺牲）；answer_relevancy 第二；context_precision / context_recall 另行优化
# concise：摘录式双目标；standard：旧版长依据（易损 faithfulness）
_raw_style = os.getenv("RAG_PROMPT_STYLE", "concise").strip().lower()
RAG_PROMPT_STYLE = _raw_style if _raw_style in ("concise", "standard") else "concise"

# ---------- 推理形态：default / cot / react / cot_react（仅改 Prompt，不增加检索轮次）----------
_raw_reasoning = os.getenv("RAG_REASONING_MODE", "default").strip().lower()
_VALID_REASONING_MODES = frozenset({"default", "cot", "react", "cot_react"})
RAG_REASONING_MODE = _raw_reasoning if _raw_reasoning in _VALID_REASONING_MODES else "default"

# 单句摘录答案时，向检索正文前后各最多扩展若干字（仍为连续原文），纳入与用户问题更易重合的词，利于 RAGAS answer_relevancy；0 关闭
RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN = _env_int("RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN", 25, 0, 120)

# ---------- 上线：明显离题短路拒答 + 引用外链（可选）----------
# 命中下列子串之一则不调 RAG/LLM，直接返回礼貌拒答（逗号分隔可覆盖默认词表）
RAG_OFF_TOPIC_ENABLED = _env_bool("RAG_OFF_TOPIC_ENABLED", True)


def _default_off_topic_keywords() -> list:
    """内置常见非人事制度问法（子串匹配，不区分大小写仅对英文部分）。"""
    return [
        "天气",
        "气温",
        "下雨",
        "下雪",
        "台风",
        "雾霾",
        "股票",
        "基金",
        "彩票",
        "笑话",
        "段子",
        "闲聊",
        "你是谁",
        "你叫什么",
        "chatgpt",
        "gpt",
        "奥运会",
        "世界杯",
        "nba",
        "今天星期",
        "现在几点",
        "黄历",
        "星座",
        "运势",
    ]


def _load_off_topic_keywords() -> list:
    raw = os.getenv("RAG_OFF_TOPIC_KEYWORDS", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return _default_off_topic_keywords()


RAG_OFF_TOPIC_KEYWORDS = _load_off_topic_keywords()
RAG_OFF_TOPIC_REPLY = os.getenv(
    "RAG_OFF_TOPIC_REPLY",
    "抱歉，我仅能根据公司已收录的人事制度文档回答问题。您的问题不在知识库范围内，我无法作答。",
).strip()
# 内部制度文档对外可访问的基址（无尾斜杠），用于 API 返回的 href；不配则前端只展示文件名与摘要
DOCS_PUBLIC_BASE_URL = os.getenv("DOCS_PUBLIC_BASE_URL", "").strip().rstrip("/")


def get_ragas_evaluator_llm():
    """
    供 RAGAS 评测使用的 Chat LLM。
    DeepSeek 仅支持 n=1；须用 ChatOpenAI 的显式参数 n=1（不可放进 model_kwargs，否则 Pydantic 校验报错）。
    """
    from langchain_community.chat_models import ChatOpenAI

    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        n=1,
    )


def get_ragas_answer_relevancy():
    """
    answer_relevancy：从答案反推「假问题」再与 user_input 做嵌入余弦相似度。
    RAGAS 默认 Prompt 为英文，DeepSeek 常反推成英文问句，而 user_input 为中文，BGE 中文向量下相似度会被压到约 0.4（与答案是否正确无关）。
    默认启用中文反推 Prompt（RAGAS_ANSWER_RELEVANCY_USE_CN_PROMPT=false 可关回库默认）。
    strictness 为每条样本反推问句次数（取均值）；中文 JSON 较稳时可设 2～3。

    NaN 常见原因：（1）反推 JSON 里 question 全为空串；（2）嵌入零向量导致余弦分母为 0。
    Robust 包装：过滤空问句、有限余弦取均值、行级重试 + 必要时 strictness=1；若仍失败则用
    「user_input ↔ 真实 response」向量余弦作兜底（有界到 [0,1]），保证不出现 NaN。
    RAGAS_ANSWER_RELEVANCY_NO_NAN_FALLBACK=false 可关兜底（此时仍可能 nan，仅调试用）。
    RAGAS_ANSWER_RELEVANCY_ROBUST=false 可关回库原生 AnswerRelevancy（易出现 NaN）。
    """
    import math

    import numpy as np
    from ragas.metrics import AnswerRelevancy
    from ragas.metrics._answer_relevance import (
        ResponseRelevanceInput,
        ResponseRelevanceOutput,
        ResponseRelevancePrompt,
        logger as _answer_relevancy_logger,
    )

    s = _env_int("RAGAS_ANSWER_RELEVANCY_STRICTNESS", 2, 1, 5)
    use_cn = _env_bool("RAGAS_ANSWER_RELEVANCY_USE_CN_PROMPT", True)
    use_robust = _env_bool("RAGAS_ANSWER_RELEVANCY_ROBUST", True)
    no_nan_fallback = _env_bool("RAGAS_ANSWER_RELEVANCY_NO_NAN_FALLBACK", True)
    row_retries = _env_int("RAGAS_ANSWER_RELEVANCY_ROW_RETRIES", 3, 1, 8)

    class ChineseResponseRelevancePrompt(ResponseRelevancePrompt):
        """与库内类同结构，仅替换 instruction / examples，使反推问句为中文、贴近 HR 问法。"""

        instruction = (
            "根据给定的【答案】，用中文写出一条可以用该答案直接、完整回答的问句；"
            "问句应包含答案中的关键政策用语或数字场景（如「免费补卡」「负激励」「旷工」等），风格接近真实员工向 HR 提问。"
            "同时判断该答案是否含糊、推脱或未针对具体问题作答：若是则 noncommittal=1，否则为 0。"
            "含糊指：不知道、不清楚、文档未载明、请联系部门等与实质内容无关的表述。"
        )
        examples = [
            (
                ResponseRelevanceInput(response="每人每月有3次免费补卡机会。"),
                ResponseRelevanceOutput(
                    question="每人每月有多少次免费补卡机会？", noncommittal=0
                ),
            ),
            (
                ResponseRelevanceInput(
                    response="迟到31分钟到60分钟以内，负激励50元/次。"
                ),
                ResponseRelevanceOutput(
                    question="迟到31分钟到60分钟以内负激励多少钱？", noncommittal=0
                ),
            ),
            (
                ResponseRelevanceInput(response="我不确定，需要咨询人事。"),
                ResponseRelevanceOutput(question="公司的考勤扣款标准是什么？", noncommittal=1),
            ),
        ]

    class RobustAnswerRelevancy(AnswerRelevancy):
        """
        主路径仍为 RAGAS：反推问句 vs user_input 的余弦（过滤空问句、有限值均值、noncommittal 规则）。
        主路径失败时用「问题↔真实答案」向量余弦兜底，避免汇总表出现 NaN（兜底分有界，不抛异常）。
        """

        def _calculate_score(self, answers, row):
            filtered = [
                a for a in answers if (getattr(a, "question", None) or "").strip()
            ]
            if not filtered:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: 反推 question 全为空，将走兜底或重试"
                )
                return float("nan")
            question = row["user_input"]
            gen_questions = [a.question for a in filtered]
            all_noncommittal = all(bool(a.noncommittal) for a in filtered)
            if all_noncommittal:
                return 0.0
            cosine_sim = self.calculate_similarity(question, gen_questions)
            arr = np.asarray(cosine_sim, dtype=float).ravel()
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: 反推余弦无非有限值，将走兜底或重试"
                )
                return float("nan")
            return float(finite.mean())

        def _fallback_relevancy_score(self, row: dict) -> float:
            """
            反推 LLM 全失败时的有限分值：user_input 与 response 的嵌入余弦，夹到 [0,1]；
            与标准 answer_relevancy 不同，仅作无 NaN 的保守替代。
            """
            if not no_nan_fallback:
                return 0.0
            u = (row.get("user_input") or "").strip()
            r = (row.get("response") or "").strip()
            if not u or not r or self.embeddings is None:
                return 0.0
            try:
                qv = np.asarray(self.embeddings.embed_query(u)).reshape(1, -1)
                rv = np.asarray(self.embeddings.embed_documents([r])).reshape(1, -1)
                nq = float(np.linalg.norm(qv))
                nr = float(np.linalg.norm(rv))
                if nq < 1e-12 or nr < 1e-12:
                    return 0.0
                c = float((np.dot(rv, qv.T).ravel()[0]) / (nq * nr))
                if not math.isfinite(c):
                    return 0.0
                return max(0.0, min(1.0, c))
            except Exception as ex:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: 兜底向量相似度异常: %s，记 0", ex
                )
                return 0.0

        async def _ascore(self, row, callbacks):
            """
            任意一步反推若抛异常（如 JSON 解析失败），RAGAS Executor 会直接记 nan；
            故对每个 generate_multiple 单独 try，并在最外层再包一层，保证永不向上抛、不返回 nan。
            """
            assert self.llm is not None
            prompt_input = ResponseRelevanceInput(response=row["response"])
            last = float("nan")
            try:
                for _ in range(row_retries):
                    try:
                        responses = await self.question_generation.generate_multiple(
                            data=prompt_input,
                            llm=self.llm,
                            callbacks=callbacks,
                            n=self.strictness,
                        )
                    except Exception as gen_ex:
                        _answer_relevancy_logger.warning(
                            "answer_relevancy: generate_multiple 异常（将重试或兜底）: %s",
                            gen_ex,
                        )
                        continue
                    last = self._calculate_score(responses, row)
                    if math.isfinite(last):
                        return last
                if self.strictness > 1:
                    try:
                        one = await self.question_generation.generate_multiple(
                            data=prompt_input,
                            llm=self.llm,
                            callbacks=callbacks,
                            n=1,
                        )
                    except Exception as gen_ex:
                        _answer_relevancy_logger.warning(
                            "answer_relevancy: strictness=1 单次反推异常: %s", gen_ex
                        )
                        one = []
                    if one:
                        last = self._calculate_score(one, row)
                        if math.isfinite(last):
                            return last
            except Exception as outer:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: _ascore 未预期异常: %s", outer, exc_info=True
                )
            fb = self._fallback_relevancy_score(row)
            if fb > 0:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: 反推未得分，已用问题↔答案向量余弦兜底: %.4f",
                    fb,
                )
            else:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: 反推未得分且兜底为 0（空文本或嵌入失败）"
                )
            return fb

        async def _single_turn_ascore(self, sample, callbacks):
            """Executor 对未捕获异常会写 nan；此处再包一层，确保任意失败都落到有限分。"""
            try:
                row = sample.to_dict()
                return await self._ascore(row, callbacks)
            except Exception as ex:
                _answer_relevancy_logger.warning(
                    "answer_relevancy: _single_turn_ascore 捕获异常，强制兜底: %s", ex
                )
                try:
                    return float(self._fallback_relevancy_score(sample.to_dict()))
                except Exception:
                    return 0.0

    qg = ChineseResponseRelevancePrompt() if use_cn else ResponseRelevancePrompt()
    if not use_robust:
        return AnswerRelevancy(strictness=s, question_generation=qg)
    return RobustAnswerRelevancy(strictness=s, question_generation=qg)


# --- 持久化路径 ---
# 项目根目录（config.py 所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(env_key: str, default_relative: str) -> str:
    """
    从环境变量读取路径；未设置则用 default_relative（相对 PROJECT_ROOT）。
    若配置为绝对路径则直接使用。
    """
    raw = os.getenv(env_key, "").strip()
    if not raw:
        raw = default_relative
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(PROJECT_ROOT, raw))


# 静态知识库文档目录（人事制度等源文件放此处；可用 .env 的 DOCS_DIR 覆盖）
DOCS_DIR = _resolve_project_path("DOCS_DIR", "docs")
# FAISS 索引目录（可用 FAISS_INDEX_PATH 设为绝对路径或相对项目根的路径）
FAISS_INDEX_PATH = _resolve_project_path("FAISS_INDEX_PATH", "faiss_index")

# 首次建库种子文档：优先 SEED_DOCUMENT_PATH（绝对或相对项目根）；否则 DOCS_DIR 下主文件名，再否则备用 txt
SEED_DOCUMENT_PATH = os.getenv("SEED_DOCUMENT_PATH", "").strip()
DEFAULT_SEED_DOCUMENT = os.getenv("DEFAULT_SEED_DOCUMENT", "人事管理流程.docx").strip()
FALLBACK_SEED_DOCUMENT = os.getenv("FALLBACK_SEED_DOCUMENT", "test_data.txt").strip()
FALLBACK_SEED_TEXT = os.getenv(
    "FALLBACK_SEED_TEXT",
    "公司规定：员工每年有10天带薪年假。病假需要提供二甲以上医院证明。",
)


def get_seed_document_path() -> str:
    """
    返回 CLI/API 首次建库时使用的文档绝对路径。
    解析顺序：SEED_DOCUMENT_PATH（若存在且为文件）→ DOCS_DIR/主文件名 → DOCS_DIR/备用名（无则创建并写入 FALLBACK_SEED_TEXT）。
    """
    if SEED_DOCUMENT_PATH:
        p = SEED_DOCUMENT_PATH
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(PROJECT_ROOT, p))
        if os.path.isfile(p):
            return p
    primary = os.path.join(DOCS_DIR, DEFAULT_SEED_DOCUMENT)
    if os.path.isfile(primary):
        return primary
    os.makedirs(DOCS_DIR, exist_ok=True)
    fallback = os.path.join(DOCS_DIR, FALLBACK_SEED_DOCUMENT)
    if os.path.isfile(fallback):
        return fallback
    with open(fallback, "w", encoding="utf-8") as f:
        f.write(FALLBACK_SEED_TEXT)
    return fallback

# --- MySQL / 生产持久化（可选，见 db/ 与 api_server）---
# 示例：mysql+pymysql://user:pass@127.0.0.1:3306/rag_db?charset=utf8mb4
# 未配置 DATABASE_URL 时，问答不落库，行为与旧版一致
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DATABASE_ENABLED = bool(DATABASE_URL)
# 启动时自动创建的默认业务用户（用于未带 X-API-Key 时的会话归属）
DEFAULT_APP_USER = os.getenv("DEFAULT_APP_USER", "app").strip() or "app"
# 若设置：写入默认用户的 api_key_hash（SHA256），请求头 X-API-Key 需与之匹配
API_SERVICE_API_KEY = os.getenv("API_SERVICE_API_KEY", "").strip()
# true 时必须在请求头携带有效 X-API-Key，否则 401
API_REQUIRE_API_KEY = _env_bool("API_REQUIRE_API_KEY", False)

# --- HTTP API（FastAPI，见 api_server.py）---
# 监听地址与端口；前端开发时端口与 Vite 错开即可
API_HOST = os.getenv("API_HOST", "0.0.0.0").strip() or "0.0.0.0"
API_PORT = _env_int("API_PORT", 8000, 1, 65535)
# 逗号分隔多个 Origin；填 * 表示允许任意来源（勿与携带 Cookie 的 credentialed 请求一起用于生产）
_api_cors_raw = os.getenv("API_CORS_ORIGINS", "*").strip()
API_CORS_ORIGINS = [x.strip() for x in _api_cors_raw.split(",") if x.strip()]


# =============================================================================
# Prompt 配置管理 - 统一管理所有 Prompt（便于后续扩展）
# =============================================================================
# 【与「Prompt 注入」的关系】
# - 用【上下文】/【问题】显式分包用户输入与检索结果，属于指令层软隔离。
# - 「铁律」约束输出形态与无答案话术，降低胡编与跑题；不能替代网关侧恶意检测。
# 【与「幻觉」的关系】
# - 要求先「推理：」再「最终答案：」便于人工与 RAGAS 拆分；最终段仍受 rag_engine 的 concise 后处理约束。
# - RAG_REASONING_MODE 切换 COT / ReAct 等说明；占位符仍为 {context}、{question}。
# =============================================================================

# 业务线开关：决定 get_current_prompt_template() 返回哪套模板（现仅 HR 一套）
RAG_SYSTEM_TYPE = os.getenv("RAG_SYSTEM_TYPE", "hr").strip().lower()

# ---------- HR Prompt：角色 + 按 RAG_REASONING_MODE 切换推理段 + 共用铁律 ----------
_HR_PROMPT_ROLE = """你是企业 HR 制度的严格问答机器人。**最终结论只回答用户明确提出的问题**，不要答非所问。
"""

_HR_REASONING_BLOCK_DEFAULT = """请先一步步思考并展示你的推理过程（简要说明依据了上下文哪些要点、如何对应到问题），再给出最终答案。

【输出格式】（必须严格遵守）
1）先写一段或多段「推理：」开头的文字（逐步推理，可换行）。
2）单独起一行，写「最终答案：」五个字加中文冒号；其**后同一行或紧随其后**只写面向用户的简洁结论，不要在该标记之前写结论。
"""

_HR_REASONING_BLOCK_COT = """请使用链式思考（Chain-of-Thought）。在写出「最终答案：」之前，必须先以「推理：」起首，并在同一段推理区内用编号步骤完整展示思路（每步一行或一句，禁止空洞套话）：
1）问题拆解：用户真正要问的结论点是什么；
2）上下文定位：【上下文】中与该点直接相关的条款或关键词（可摘录≤20字原文作锚点）；
3）条件核对：是否存在适用前提、例外或数字区间；
4）自检：结论是否仅来自【上下文】字面，无外延臆测。

【输出格式】（必须严格遵守）
1）推理区整体以「推理：」开头，其内包含上述编号步骤（可略增子步骤，但须保持编号清晰）。
2）单独起一行写「最终答案：」加中文冒号；其后只写面向用户的简洁结论，不要在该标记之前写结论。
"""

_HR_REASONING_BLOCK_REACT = """请使用 ReAct 风格：在已给出的【上下文】内进行「思考 → 行动 → 观察」循环（至多 2 轮即可，勿冗长）。禁止编造未出现在【上下文】中的条文或数字；「观察」必须是从【上下文】可逐字支持的要点。

每轮请严格使用下列行首标签（便于阅读）：
思考：……
行动：在【上下文】中查找/对照的具体关键词或条款位置说明（可带≤30字引号摘录作为定位锚点）……
观察：基于【上下文】摘录得到的客观要点（勿在此步骤做最终对用户答复的完整结论文）……

完成至多 2 轮后，用一行「思考：」简要汇总是否足以作答，然后**必须**另起一行写「最终答案：」（五个字加中文冒号）及结论。

【输出格式】（必须严格遵守）
1）「最终答案：」必须单独成行；其前为 ReAct 循环与最后一行「思考：」汇总。
2）「最终答案：」之后只写面向用户的简洁结论。
"""

_HR_REASONING_BLOCK_COT_REACT = """请结合 ReAct 与链式思考（COT）：
第一阶段（ReAct，至多 2 轮）：每轮包含
思考：……
行动：……（仅在【上下文】内检索/对照，可含短引号摘录）
观察：……（仅客观摘录，不做最终用户口径结论）
第二阶段（COT）：以「推理：」起首，用编号步骤 1）2）3）4）对应「问题拆解 → 证据归纳 → 条件与例外 → 与【上下文】字面一致性自检」。
最后单独一行「最终答案：」加中文冒号，其后只写简洁结论。

【输出格式】（必须严格遵守）
1）顺序固定：ReAct 循环 →「推理：」+ 编号 COT →「最终答案：」。
2）「最终答案：」之前不得写出面向用户的完整结论文（仅可在「观察」「推理」中写分析性语句）。
"""

_HR_PROMPT_IRON_AND_SLOTS = """
【铁律】（仅约束「最终答案：」之后的内容）
- 最终答案第一句必须直接包含问题的关键词或量词。
- 若问题中出现业务主题词（如「免费补卡」「负激励」「旷工」等），结论中须保留与上下文一致的同一表述，勿仅用数字或「按规定」等代称带过（仍不得编造上下文中没有的词）。
- 只输出用户问到的内容，绝对不要包含其他规则。
- 禁止任何套话。
- 如果没有答案，「最终答案：」后只写：根据提供的文档，无法找到相关信息
- 若问题明显与员工制度无关（如天气、新闻、娱乐、闲聊、生活常识），「最终答案：」后写：抱歉，我仅能根据公司人事制度文档回答问题；该问题不在知识库范围内。

【上下文】
{context}

【问题】
{question}"""


def _hr_prompt_body_for_reasoning_mode(mode: str) -> str:
    """按 RAG_REASONING_MODE 选取推理说明段，拼成完整 HR 模板。"""
    block_map = {
        "default": _HR_REASONING_BLOCK_DEFAULT,
        "cot": _HR_REASONING_BLOCK_COT,
        "react": _HR_REASONING_BLOCK_REACT,
        "cot_react": _HR_REASONING_BLOCK_COT_REACT,
    }
    mid = block_map.get(mode, _HR_REASONING_BLOCK_DEFAULT)
    return _HR_PROMPT_ROLE + mid + _HR_PROMPT_IRON_AND_SLOTS


# 与 RAG_REASONING_MODE=default 时 get_current_prompt_template() 全文一致（便于对照）
HR_PROMPT_TEMPLATE = _hr_prompt_body_for_reasoning_mode("default")


def get_current_prompt_template():
    """返回当前 RAG_SYSTEM_TYPE 与 RAG_REASONING_MODE 下的 Prompt（须含 {context} 与 {question}）。"""
    if RAG_SYSTEM_TYPE == "hr":
        return _hr_prompt_body_for_reasoning_mode(RAG_REASONING_MODE)
    return _hr_prompt_body_for_reasoning_mode(RAG_REASONING_MODE)


print(
    f"[INFO] 当前系统类型: {RAG_SYSTEM_TYPE}，推理模式: {RAG_REASONING_MODE}，Prompt 模板已加载"
)
