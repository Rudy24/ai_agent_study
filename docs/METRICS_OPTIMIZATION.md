# RAG 评测指标优化说明

本文档归纳本项目中 **RAGAS 指标**（尤其是 **faithfulness**、**answer_relevancy**）的优化目标、问题成因、已采取手段与配置入口，便于复现调参与后续迭代。

---

## 1. 优化目标与优先级

项目在 `config.py` 中明确评测调优顺序：

1. **faithfulness（答案忠实度）**：答案是否可被检索上下文支撑，优先保证，不轻易为抬高其它指标而牺牲。
2. **answer_relevancy（答案相关性）**：答案是否直接回应用户问题，次之。
3. **context_precision / context_recall**：检索层指标，随检索参数、切块与重排单独调。

离线汇总时，`ragas/eval_n_questions.py` 等脚本会额外打印 **「主指标(忠实+相关)」**：对 `faithfulness` 与 `answer_relevancy` 两列分别求均值后再取平均，作为一轮实验的优先对比口径。

---

## 2. 评测链路如何保证「评的是我们想评的」

若评测用的 `answer` 含大段「推理」前缀，或 `contexts` 与生成时实际依据的正文不一致，指标会偏离真实业务表现。本项目做了以下对齐：

| 环节 | 做法 | 代码位置 |
|------|------|----------|
| 送入 RAGAS 的答案 | 使用 `answer_only`（仅「最终答案：」之后段落），避免推理前缀干扰 faithfulness / answer_relevancy | `ragas/eval_n_questions.py` → `run_rag_on_questions` |
| 送入 RAGAS 的 contexts | 每条引用用 `document_context_text_for_eval(d)`：与线上 `_build_context` 一致，**优先 `metadata.parent_content`，否则子块 `page_content`** | `rag_engine.py`、`eval_n_questions.py` |
| 生成侧 context 拼接 | `_merged_raw_doc_text` 同时并入 `parent_content` 与子块正文，供 concise 后处理做「分句是否在上下文中」校验；若漏掉父块，易出现答案字面不在 blob 中，**answer_relevancy 被动压低** | `rag_engine.py` |

**结论**：优化指标时，既要改 Prompt/检索，也要保证 **评测构造的 answer 与 contexts 与线上一致**，否则会出现「线上观感好、分数异常」的假问题。

---

## 3. 生成侧：concise 模式与 HR Prompt（服务 faithfulness + relevancy）

### 3.1 `RAG_PROMPT_STYLE=concise`（默认）

- 模型输出经 `format_answer_with_reasoning`：先拆「推理」与「最终答案」，**仅对最终段**做 `apply_concise_answer_postprocess`。
- 后处理流水线（见 `rag_engine.py`）：
  - **归一化空白**：减少 faithfulness 因换行/格式被拆成多句而误判。
  - **`clip_concise_clauses_by_context`**：多「；」分句时，去掉在检索拼接正文中**无连续字面**支撑的分句，抑制杜撰分句拉低 faithfulness。
  - **`trim_redundant_clauses_for_multi_question`**：用户有多个「？」时，裁掉多余分句，减少跑题摘录，有利于 **answer_relevancy**（保留句仍须为原文子串）。
  - **`expand_single_clause_for_relevancy` / `RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN`**：在仍为连续原文的前提下向两侧扩窗，仅当与用户问题的字符重合**严格增加**才采纳，避免无意义加长带来的忠实度风险。

### 3.2 HR 主模板 `HR_PROMPT_TEMPLATE`

- 要求先「推理」再「最终答案：」，便于拆分与人工检查。
- **铁律**约束最终答案形态：须扣题、保留关键政策用语、无信息时固定话术等，从源头减少胡编与答非所问。
- 与「Prompt 注入」关系及安全边界说明见 `config.py` 中 Prompt 区块注释及 `docs/ARCHITECTURE.md` 第 4.1 节。

---

## 4. RAGAS 侧：answer_relevancy 专项（中文 + 稳健实现）

RAGAS 默认 **answer_relevancy** 用英文反推「假问题」，再与 `user_input` 做嵌入相似度。中文场景下 DeepSeek 常反推成英文问句，而用户问题是中文，在 **BGE 中文向量** 下相似度会被压到约 0.4 量级，**与答案是否正确弱相关**，导致指标失真。

### 4.1 中文反推 Prompt

- `config.get_ragas_answer_relevancy()` 中 `ChineseResponseRelevancePrompt`：instruction 与 examples 改为 **中文 HR 风格**，使反推问句与 `user_input` 处于同一语言空间。
- 环境变量：`RAGAS_ANSWER_RELEVANCY_USE_CN_PROMPT`（默认 true，设为 false 可回库默认英文行为）。

### 4.2 RobustAnswerRelevancy

- **问题**：反推 JSON 异常、空 question、嵌入零向量等会导致 RAGAS 汇总出现 **NaN**。
- **手段**（`config.py` 内嵌类）：
  - `_calculate_score`：过滤空问句；全非 `noncommittal` 时对有效余弦取均值。
  - 行级 **`RAGAS_ANSWER_RELEVANCY_ROW_RETRIES`** 重试 `generate_multiple`；失败则尝试 **strictness=1** 单次反推。
  - 仍失败时用 **`user_input` ↔ 真实 `response` 的向量余弦** 作兜底，分值夹到 `[0,1]`，保证表格中不出现 NaN（`RAGAS_ANSWER_RELEVANCY_NO_NAN_FALLBACK=false` 可关闭兜底，仅建议调试）。
- 环境变量：`RAGAS_ANSWER_RELEVANCY_ROBUST`（默认 true）、`RAGAS_ANSWER_RELEVANCY_STRICTNESS`（反推条数，默认 2；不稳时可试 1）、`RAGAS_ANSWER_RELEVANCY_ROW_RETRIES`。

### 4.3 评测用 LLM

- `get_ragas_evaluator_llm()`：DeepSeek 要求 **`n=1`** 显式传入，不可放入 `model_kwargs`，否则 Pydantic 校验失败。

---

## 5. 检索与重排参数（影响 context_* 与间接影响 faithfulness）

下列变量在 `config.py` / `.env` 中配置，通过改变「进入 Prompt 的文档质量与数量」影响评测：

| 变量 | 作用简述 |
|------|----------|
| `SEARCH_K` | 向量路与 BM25 路各自召回条数上限 |
| `RERANK_POOL_SIZE` / `RERANK_TOP_N` | 重排前候选池与最终进上下文的条数 |
| `RERANK_MAX_PASSAGE_CHARS` / `RERANK_BATCH_SIZE` | CrossEncoder 输入截断与批大小（偏延迟与稳定性，间接影响排序质量） |
| `USE_RERANKER` | 是否启用 CrossEncoder 精排 |

**经验向建议**（需结合本库文档与压测再定）：context_recall 偏低可适度增大召回或池子；faithfulness 偏低优先查 Prompt 与 concise 裁剪是否过宽、检索是否引入噪声，而非单纯加长篇上下文。

---

## 6. 离线评测脚本与结果解读

- **推荐入口**：`python ragas/eval_n_questions.py --num N`（问题集默认 `ragas/data/hr_eval_questions.json`）。
- 其它脚本（如 `eval_ragas.py`、`eval_hr_30.py`、`auto_eval_pipeline.py`）同样使用 `get_ragas_answer_relevancy()` 时，与上述 answer_relevancy 行为一致。
- 结果表中对异常单元格做了 **`_scalar_metric_value`** 归一化，避免把 numpy 标量或空列表误判为 N/A。

**迭代流程建议**：

1. 固定索引与 `.env`（或记录 commit），跑一轮得到四列指标 + 主指标。
2. 若 **faithfulness** 低：抽查该条 `contexts` 与 `answer` 是否字面一致、是否被后处理误删、Prompt 是否诱导扩写。
3. 若 **answer_relevancy** 低：看问题是否多问了未答的点；可调 `RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN`、Prompt 铁律；并确认评测用的是 `answer_only` 与 `document_context_text_for_eval`。
4. 若 **answer_relevancy** 大量异常或日志大量兜底：检查 `RAGAS_ANSWER_RELEVANCY_*` 与 DeepSeek 反推 JSON 是否稳定。

---

## 7. 相关文件索引

| 文件 | 与指标的关系 |
|------|----------------|
| `config.py` | `RAG_PROMPT_STYLE`、`RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN`、`get_ragas_answer_relevancy`、`get_ragas_evaluator_llm`、`HR_PROMPT_TEMPLATE` |
| `rag_engine.py` | concise 后处理、`_merged_raw_doc_text`、`document_context_text_for_eval`、`format_answer_with_reasoning` |
| `ragas/eval_n_questions.py` | 构造 `answer` / `contexts`、指标列表、主指标汇总 |
| `docs/ARCHITECTURE.md` | 第 4 节 RAG 机制、幻觉控制、记忆边界 |
| `ragas/README_EVAL.md` | RAGAS 使用与通用排错思路 |

---

## 8. 与「性能优化」的区分

为降低 API 延迟引入的 `RAG_LOW_LATENCY_MODE`、`RAG_LLM_MAX_TOKENS`、收紧 `RERANK_*` 等，可能略微影响 context 覆盖或答案长度，进而影响评测分数。建议 **对比实验** 时单独记录是否开启低开销模式，避免把「延迟优化」与「指标优化」混在一轮结论里。

---

*文档随代码变更更新；若新增指标或更换评测库版本，请同步修订本节并检查 `get_ragas_answer_relevancy` 与 RAGAS API 是否仍兼容。*
