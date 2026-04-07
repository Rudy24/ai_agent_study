# RAGAS 评估系统使用指南

## 简介

RAGAS（Retrieval-Augmented Generation Assessment Suite）是一个专门用于评估 RAG 系统性能的框架。本评估系统可以帮助你定量分析 RAG 系统的检索质量和答案生成质量。

## 评估指标说明

| 指标 | 英文名称 | 含义 | 理想值 |
|------|---------|------|--------|
| 上下文精确度 | Context Precision | 检索到的文档中有多少与答案相关 | 1.0 |
| 上下文召回率 | Context Recall | 答案中信息有多少来自检索到的文档 | 1.0 |
| 上下文相关性 | Context Relevancy | 检索到的文档与问题的相关程度 | 1.0 |
| 答案忠实度 | Faithfulness | 答案是否基于检索到的文档（无幻觉） | 1.0 |
| 答案相关性 | Answer Relevancy | 答案是否直接回答了问题 | 1.0 |
| 答案正确性 | Answer Correctness | 答案与标准答案的匹配程度 | 1.0 |

## 快速开始

### 1. 安装依赖

```bash
pip install ragas pandas openpyxl
```

### 2. 创建评估数据集

**方式一：手动收集（推荐，质量更高）**

```bash
python generate_eval_dataset.py --mode manual --output eval_dataset.json
```

然后按提示：
1. 输入要测试的问题
2. 查看系统回答
3. 输入你认为正确的标准答案
4. 输入 `save` 保存退出

**方式二：自动生成（快速生成大量样本）**

```bash
python generate_eval_dataset.py --mode auto --output eval_dataset.json --num 20
```

### 3. 运行评估

```bash
python eval_ragas.py --input eval_dataset.json --output eval_results.xlsx
```

评估完成后会生成 Excel 报告，包含：
- **RAGAS Results** 工作表：每条样本的详细评分
- **Statistics** 工作表：各项指标的平均值、标准差等统计信息

### 4. 查看结果

控制台会输出评分摘要：

```
============================================================
RAGAS 评估结果摘要
============================================================
  context_precision           : 0.850
  context_recall              : 0.780
  context_relevancy           : 0.920
  faithfulness                : 0.880
  answer_relevancy            : 0.910
  answer_correctness          : 0.820
============================================================
```

## 评估数据集格式

评估数据集是 JSON 格式，每个样本包含以下字段：

```json
{
  "question": "用户提问",
  "answer": "RAG系统生成的答案",
  "contexts": ["检索到的文档片段1", "检索到的文档片段2"],
  "ground_truth": "标准答案（人工标注的正确答案）"
}
```

## 评估报告解读

### 低上下文召回率（Context Recall < 0.7）
- **问题**：检索器没有召回足够的相关文档
- **解决方案**：
  - 增加 `SEARCH_K` 参数，提高召回数量
  - 优化文本切分策略（调整 chunk_size/overlap）
  - 检查嵌入模型是否适合中文

### 低忠实度（Faithfulness < 0.7）
- **问题**：答案中包含检索文档以外的信息（幻觉）
- **解决方案**：
  - 在 Prompt 中明确要求"只根据上下文回答"
  - 调整 LLM temperature 为更低值
  - 添加上下文压缩或重排序

### 低答案相关性（Answer Relevancy < 0.7）
- **问题**：答案没有直接回答问题
- **解决方案**：
  - 优化 Prompt 模板，要求"简练、直接回答问题"
  - 检查检索文档是否与问题相关
  - 调整检索权重（BM25 vs 向量）

## 常见问题

### Q: 评估过程很慢？

RAGAS 需要调用 LLM API 多次（每个样本 × 多个指标），建议：
- 评估样本数量控制在 20-50 条
- 使用响应更快的 API
- 在 `.env` 中配置更快的 LLM 模型

### Q: 可以离线评估吗？

可以！修改 `eval_ragas.py` 中的 `setup_evaluator()` 函数，使用本地 LLM（如通过 Ollama）。

### Q: 评估结果不稳定？

LLM 评估有一定随机性，建议：
- 多次评估取平均值
- 增加评估样本数量
- 使用 temperature=0 降低随机性

## 进阶使用

### 批量对比不同配置

创建 `eval_compare.py`：

```python
# 比较不同 chunk_size 的效果
for chunk_size in [300, 500, 800]:
    # 重新构建索引
    # 运行评估
    # 记录结果
    pass
```

### 集成到 CI/CD

```bash
# 在部署前自动运行评估
python eval_ragas.py --input eval_dataset.json
# 如果平均分低于阈值，阻止部署
```

## 参考资源

- RAGAS 官方文档：https://docs.ragas.io/
- RAGAS GitHub：https://github.com/explodinggradients/ragas
