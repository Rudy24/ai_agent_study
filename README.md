# RAG 客服系统（本地化版）

基于 LangChain + FAISS + 本地中文向量模型 + DeepSeek API 的检索增强生成（RAG）客服 Demo。

**架构与业务逻辑说明**见：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（含 **Prompt 注入防护、上下文压缩、Chunk 策略、幻觉控制、记忆机制** 等与代码一一对应的说明）。

## 核心机制摘要

| 主题 | 行为概要 |
|------|----------|
| **Prompt 注入** | 模板分隔「上下文 / 问题」+ 铁律 + `temperature=0`；**非**独立安全模块，生产需网关防护。 |
| **长文本压缩** | 可选 `LLMChainExtractor` 对检索结果再摘录后再送入生成（见 `rag_engine.py`）。 |
| **Chunk** | 父子切块：子块入 FAISS，生成时优先用父块 `parent_content` 拼上下文（见 `knowledge_base.py`）。 |
| **幻觉控制** | 混合检索 + 可选重排；`concise` 模式下按检索正文做分句过滤与多问问句对齐（见 `rag_engine.py`）。 |
| **记忆** | MySQL 存会话供 UI 与历史 API；**当前不**把历史轮次注入 LLM Prompt（每轮仅当前问 + 检索上下文）。 |

## 项目特点

- **完全离线运行**：支持将 `bge-small-zh-v1.5` 模型下载到本地，后续无需联网
- **混合检索**：向量语义检索 + BM25 关键词检索，提升召回率
- **上下文压缩**：使用 LLM 从检索结果中提取最相关片段
- **中文优化**：使用 BAAI 的中文嵌入模型，对中文文档效果好

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型（仅需一次，可选但强烈推荐）

**方式 A：使用国内镜像（推荐国内用户）**
```bash
set HF_ENDPOINT=https://hf-mirror.com
python download_model.py
```

**方式 B：使用 HuggingFace 官方源**
```bash
python download_model.py
```

下载完成后，模型将保存在 `models/bge-small-zh-v1.5/` 目录，`.env` 会自动配置 `LOCAL_MODEL_PATH`。

### 3. 配置 API 密钥

编辑 `.env` 文件：
```bash
DEEPSEEK_API_KEY=your_api_key_here
```

在 [DeepSeek 开放平台](https://platform.deepseek.com) 申请 API 密钥。

### 4. 运行项目

```bash
python main.py
```

首次运行会构建知识库索引（FAISS），后续直接从本地加载。

### 上线前检查（延迟 / 拒答 / 引用）

| 项 | 说明 |
|----|------|
| **延迟** | 典型一问的端到端时间取决于本机 **GPU（嵌入+Rerank）**、DeepSeek 网络 RTT 与是否流式首包；可在浏览器开发者工具里看 **chat/stream** 从请求到 `t=final` 的耗时，或用秒表粗测 3 个制度问题。已启用 Rerank+GPU 时一般明显快于纯 CPU。 |
| **知识库外问题** | 对「天气、股票、笑话」等 **明显离题** 问句，服务端会 **短路** 返回礼貌拒答（不调检索/LLM），词表见 `config.RAG_OFF_TOPIC_KEYWORDS`，可用环境变量 `RAG_OFF_TOPIC_KEYWORDS` 追加（逗号分隔）。**人事制度内但文档未写** 的问题仍依赖 Prompt 输出「根据提供的文档，无法找到相关信息」。 |
| **引用来源** | `POST /api/chat` 与 **SSE `t=final`** 的 JSON 中均含 **`sources`**（`label`、`preview`、可选 **`href`**）。前端已在气泡下展示「参考来源」。若需 **可点击下载/预览链接**，在 `.env` 配置 **`DOCS_PUBLIC_BASE_URL`**（内网文档站点基址，无尾斜杠）；子块 **`metadata.source`** 需在 **重建索引** 后才有完整文件名（见 `knowledge_base` 写入逻辑）。 |

## RAGAS 质量评测

使用 [RAGAS](https://docs.ragas.io/) 对当前 RAG 链路做离线打分：先按测试集跑 `RAGEngine.ask` 得到答案与 `source_documents`，再计算 **context_precision**、**context_recall**、**faithfulness**、**answer_relevancy**（控制台会额外汇总「主指标」：忠实度与答案相关性优先）。

### 环境与依赖

- 在项目根目录、**与线上一致的虚拟环境**下执行（需能 `import config`、加载 FAISS 与 `.env` 中的 `DEEPSEEK_API_KEY`）。
- `requirements.txt` 已包含 `ragas`、`pandas`、`openpyxl`；若缺包可执行：`pip install -r requirements.txt`。

### 推荐入口（项目根目录）

| 命令 | 说明 |
|------|------|
| `python run_eval.py` | 默认评测 **3** 条（内部调用 `ragas/quick_eval.py` → `eval_n_questions.py`） |
| `python run_eval.py 10` | 评测 10 条 |
| `python run_eval.py --all` | 评测问题集中的全部条目 |
| `python ragas/quick_eval.py --help` | 快捷脚本参数（`--input`、`--output`、`--save-dataset` 等） |

等价地也可直接运行：

```bash
python ragas/eval_n_questions.py --num 5
python ragas/eval_n_questions.py --all
python ragas/eval_n_questions.py --num 3 --output my_eval.xlsx --save-dataset
```

### 测试数据与输出位置

- 默认问题集：`ragas/data/hr_eval_questions.json`（`question` + `ground_truth`）；可通过 `--input` 指定其它 JSON。
- 结果默认写入 **`ragas/results/`**（相对 `--output` 文件名会解析到该目录）：Excel（`.xlsx`）及同基名的 JSON 摘要。

### 说明与排错

- RAGAS 会多次调用评测用 LLM（与 DeepSeek 配置相关），**耗时长、会产生 API 费用**，建议先用少量 `--num` 试跑。
- 指标解读、自建数据集格式、调参建议等见 **`ragas/README_EVAL.md`**（含 `generate_eval_dataset.py` / `eval_ragas.py` 等扩展流程）。

## 项目结构

```
my_rag_system/
├── .env                      # 环境变量配置（API密钥、设备、模型路径）
├── requirements.txt          # Python依赖
├── run_eval.py               # RAGAS 快捷入口（默认 3 条）
├── config.py                 # 全局配置
├── knowledge_base.py         # 文档处理与向量库构建
├── rag_engine.py             # 检索与问答逻辑
├── main.py                   # 启动入口
├── api_server.py             # FastAPI 服务入口
├── download_model.py         # 模型下载脚本
├── tests/                    # 测试与压测
│   └── concurrency/          # /api/chat 并发压测（标准库，见 run_concurrent_chat.py）
├── ragas/                    # RAGAS 评测脚本与数据
│   ├── data/                 # 评测问题集（如 hr_eval_questions.json）
│   ├── results/              # 评测输出（Excel/JSON）
│   ├── eval_n_questions.py   # 主流程：跑 RAG + RAGAS 四指标
│   ├── quick_eval.py         # 参数转发到 eval_n_questions
│   └── README_EVAL.md        # 评测扩展说明与指标详解
├── models/                   # 本地模型目录（由 download_model.py 创建）
│   └── bge-small-zh-v1.5/    # 嵌入模型文件
├── docs/                     # 静态源文档目录（可用 DOCS_DIR 配置）
├── faiss_index/              # FAISS 向量索引（自动生成，路径可配）
└── README.md                 # 本文件
```

## 核心配置说明

### `.env` 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 必填 |
| `EMBEDDING_DEVICE` | 嵌入模型运行设备 | `auto`（自动检测 GPU） |
| `HF_ENDPOINT` | HuggingFace 镜像地址 | 空（官方源） |
| `LOCAL_MODEL_PATH` | 本地模型路径 | `models/bge-small-zh-v1.5` |
| `DOCS_DIR` | 静态知识库文档目录（相对项目根或绝对路径） | `docs` |
| `FAISS_INDEX_PATH` | FAISS 索引目录 | `faiss_index` |
| `SEED_DOCUMENT_PATH` | 首次建库指定单文件（优先于下面两项） | 空 |
| `DEFAULT_SEED_DOCUMENT` | 在 `DOCS_DIR` 下的主种子文件名 | `人事管理流程.docx` |
| `FALLBACK_SEED_DOCUMENT` | 主文件不存在时使用的备用文件名 | `test_data.txt` |
| `DATABASE_URL` | MySQL 连接串（SQLAlchemy），不配则不落库 | 空 |
| `API_SERVICE_API_KEY` | 与请求头 `X-API-Key` 对应的明文（库中存 SHA256） | 空 |
| `API_REQUIRE_API_KEY` | 是否强制要求 `X-API-Key` | `false` |
| `DEFAULT_APP_USER` | 未带 Key 时会话归属的默认用户名 | `app` |
| `RAG_PROMPT_STYLE` | `concise`（摘录后处理，利评测忠实度）或 `standard` | `concise` |
| `RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN` | 单句答案在检索正文中前后扩窗字数上限（0 关闭） | `25` |
| `USE_RERANKER` | 是否启用本地 CrossEncoder 重排 | `true` |
| `RERANK_POOL_SIZE` / `RERANK_TOP_N` | 重排前候选池大小 / 进入 Prompt 的条数 | `30` / `5` |

### 生产级：MySQL 持久化（向量仍用 FAISS）

1. 创建数据库：`CREATE DATABASE rag_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`
2. `.env` 配置 `DATABASE_URL=mysql+pymysql://用户:密码@主机:3306/rag_db?charset=utf8mb4`
3. 启动 `api_server` 会自动建表 `rag_users` / `rag_conversations` / `rag_messages` 并创建默认用户
4. 生产建议设置 `API_SERVICE_API_KEY` 与 `API_REQUIRE_API_KEY=true`，前端配置 `VITE_API_KEY`
5. 检索向量仍在 `FAISS_INDEX_PATH` 目录，**无需**把 FAISS 文件迁入 MySQL

### 设备配置

```bash
# 强制使用 GPU（需安装 CUDA 版 PyTorch）
EMBEDDING_DEVICE=cuda

# 强制使用 CPU
EMBEDDING_DEVICE=cpu

# 自动检测（有 NVIDIA 显卡则用 GPU，否则 CPU）
EMBEDDING_DEVICE=auto
```

## 常见问题

### Q: 如何完全离线运行？

1. 运行 `python download_model.py` 下载模型到本地
2. 确保已构建 FAISS 索引（运行过一次 `main.py` 并成功处理文档）
3. 之后无需联网，直接 `python main.py` 即可

### Q: 模型下载很慢或失败？

**国内用户建议使用镜像：**
```bash
set HF_ENDPOINT=https://hf-mirror.com
python download_model.py
```

**手动下载：**
访问 https://hf-mirror.com/BAAI/bge-small-zh-v1.5 点击「下载」按钮，解压后放到 `models/bge-small-zh-v1.5/` 目录。

### Q: 如何更换知识库文档？

1. 将新文件放入 `docs/`（或 `.env` 中的 `DOCS_DIR`），并通过 `SEED_DOCUMENT_PATH` 或 `DEFAULT_SEED_DOCUMENT` 指定首次建库文件。
2. 删除 `faiss_index/`（或 `FAISS_INDEX_PATH` 指向的目录）后重新运行 `main.py` / `api_server.py`，会按种子文档重建索引。

### Q: Windows 中文显示乱码？

在 CMD 或 PowerShell 执行：
```bash
chcp 65001
```

或使用 Git Bash 终端。

## 源码阅读

核心逻辑集中在 `config.py`（Prompt 与超参）、`knowledge_base.py`（切块与索引）、`rag_engine.py`（检索/压缩/生成/后处理）、`api_server.py`（HTTP 与落库）、`db/*`（会话表）；上述文件已补充**中文模块说明与关键步骤注释**，细节以代码为准并与 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 第 4 节对照阅读。

## 技术栈

- **框架**: LangChain
- **向量库**: FAISS
- **嵌入模型**: BAAI/bge-small-zh-v1.5（本地化）
- **对话模型**: DeepSeek API（需联网）
- **检索**: 向量检索 + BM25 混合
- **压缩**: LLMChainExtractor 上下文压缩

## License

仅供学习参考使用。
