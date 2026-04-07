"""
全局配置模块（config.py）
------------------------
作用：集中管理嵌入模型名、DeepSeek 对话参数、检索条数、FAISS 索引路径。
数据流：启动时 load_dotenv() 读取 .env；各模块 import 本文件中的常量即可。
"""
import os  # 读取环境变量
from dotenv import load_dotenv  # 加载 .env 到 os.environ

# 将项目根目录下的 .env 合并进环境（若不存在则静默跳过）
load_dotenv()

# Windows / 多原生库：PyTorch、NumPy、FAISS、rank-bm25 各带 OpenMP 时易冲突导致段错误；须在首次 import torch / langchain 之前生效
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 强制 NumExpr 也单线程，减少 Windows 多进程/多 import 下的段错误
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- 嵌入模型（本地 HuggingFace，无需为向量付 API 费）---
# 可选模型（按性能排序）：
# 1. BAAI/bge-large-zh-v1.5 - 性能最好，需要更多内存
# 2. BAAI/bge-base-zh-v1.5 - 平衡性能和资源
# 3. BAAI/bge-small-zh-v1.5 - 轻量级，资源占用小
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 升级为大模型，提高检索准确性
# 嵌入推理设备：环境变量 EMBEDDING_DEVICE，可选 auto / cuda / cpu（auto 时检测到 CUDA 则用 GPU）
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
# HuggingFace 国内镜像 endpoint（解决 huggingface.co 连接超时），如 https://hf-mirror.com
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "").strip().rstrip("/")
# 本地模型缓存目录（若已手动下载模型，可直接指向本地路径，跳过网络下载）
# 默认指向项目目录下的 models/bge-small-zh-v1.5，方便统一管理
_DEFAULT_LOCAL_MODEL = os.path.join(os.path.dirname(__file__), "models", "bge-small-zh-v1.5")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", _DEFAULT_LOCAL_MODEL).strip()

# --- DeepSeek 对话 LLM（OpenAI 兼容接口）---
# API 密钥来自 .env 的 DEEPSEEK_API_KEY，勿写死在代码里
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 模型名按 DeepSeek 开放平台文档填写
DEEPSEEK_MODEL = "deepseek-chat"
# 基础 URL 需带 /v1，与 LangChain OpenAI 封装的路径约定一致
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# --- 检索与重排序相关数量 ---
# SEARCH_K：向量检索与 BM25 各自召回的文档条数（增大到20，提高召回率）
SEARCH_K = 25
# RERANK_TOP_N：最终进入 LLM 的文档条数（限制为5，只保留最相关文档）
RERANK_TOP_N = 5
# 以下两项为预留阈值：当前 rag_engine 检索路径未读取（过滤主要靠 SEARCH_K 与重排 Top-N）
SIMILARITY_THRESHOLD = 0.6
RELEVANCE_THRESHOLD = 0.5

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


def _env_bool(name: str, default: bool) -> bool:
    """解析环境变量布尔值：空串用 default，0/false/no/off 为 False。"""
    v = os.getenv(name, "").strip().lower()
    if v == "":
        return default
    return v not in ("0", "false", "no", "off", "n")


USE_RERANKER = _env_bool("USE_RERANKER", True)
# 重排前合并向量+BM25 候选的最大去重条数（越大越稳、越慢）
RERANK_POOL_SIZE = int(os.getenv("RERANK_POOL_SIZE", "30"))
# CrossEncoder.predict 批次大小（显存不足可调小）
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))

# RAG 评测与调优目标：faithfulness 第一（不可牺牲）；answer_relevancy 第二；context_precision / context_recall 另行优化
# concise：摘录式双目标；standard：旧版长依据（易损 faithfulness）
_raw_style = os.getenv("RAG_PROMPT_STYLE", "concise").strip().lower()
RAG_PROMPT_STYLE = _raw_style if _raw_style in ("concise", "standard") else "concise"


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


# 单句摘录答案时，向检索正文前后各最多扩展若干字（仍为连续原文），纳入与用户问题更易重合的词，利于 RAGAS answer_relevancy；0 关闭
RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN = _env_int("RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN", 25, 0, 120)


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
    answer_relevancy：从答案反推问题再与 user_input 做嵌入相似度；反推结果须解析为含 question 字段的 JSON。
    strictness>1 时 RAGAS 会多次调用同一结构化 prompt，DeepSeek 任一次返回非合法 JSON 或空 question，该样本会得到 NaN/N/A。
    默认 strictness=1 最稳；仅在反推稳定时可设环境变量为 2～3 试抬均分。
    """
    from ragas.metrics import AnswerRelevancy

    s = _env_int("RAGAS_ANSWER_RELEVANCY_STRICTNESS", 1, 1, 5)
    return AnswerRelevancy(strictness=s)


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
# =============================================================================

# 业务线开关：决定 get_current_prompt_template() 返回哪套模板（现仅 HR 一套）
RAG_SYSTEM_TYPE = os.getenv("RAG_SYSTEM_TYPE", "hr").strip().lower()

# HR 制度主模板：input_variables 必须为 context、question，与 RAGEngine.optimized_prompt 一致
HR_PROMPT_TEMPLATE = """你是企业 HR 制度的严格问答机器人。**最终结论只回答用户明确提出的问题**，不要答非所问。

请先一步步思考并展示你的推理过程（简要说明依据了上下文哪些要点、如何对应到问题），再给出最终答案。

【输出格式】（必须严格遵守）
1）先写一段或多段「推理：」开头的文字（逐步推理，可换行）。
2）单独起一行，写「最终答案：」五个字加中文冒号；其**后同一行或紧随其后**只写面向用户的简洁结论，不要在该标记之前写结论。

【铁律】（仅约束「最终答案：」之后的内容）
- 最终答案第一句必须直接包含问题的关键词或量词。
- 只输出用户问到的内容，绝对不要包含其他规则。
- 禁止任何套话。
- 如果没有答案，「最终答案：」后只写：根据提供的文档，无法找到相关信息

【上下文】
{context}

【问题】
{question}"""


def get_current_prompt_template():
    """返回当前 RAG_SYSTEM_TYPE 对应的 Prompt 字符串（须含 {context} 与 {question} 占位符）。"""
    if RAG_SYSTEM_TYPE == "hr":
        return HR_PROMPT_TEMPLATE
    return HR_PROMPT_TEMPLATE


print(f"[INFO] 当前系统类型: {RAG_SYSTEM_TYPE}，Prompt 模板已加载")
