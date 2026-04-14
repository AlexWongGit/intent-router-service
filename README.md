# Intent Router Service

一个面向企业级 RAG / Agent 系统的意图识别与路由服务。
训练数据集中在 PLM（Product Lifecycle Management）领域，用于展示垂域语义理解能力与工程化落地能力。

## 项目简介

Intent Router Service 是一个独立部署的意图识别服务，用于在多工具、多能力系统中，对用户请求进行语义理解与路由决策。

系统采用 hybrid 架构：

- Embedding Retrieval（语义召回）
- Classifier（判别模型）
- Policy（策略决策 + 澄清机制）

## 核心能力
	•	多意图识别（支持 10+ intent）
	•	模糊语义自动澄清（clarification）
	•	高风险意图冲突控制（create/update/workflow）
	•	可解释性输出（debug 模式）
	•	本地模型加载（支持离线部署）
	•	错误样本驱动优化（持续提升效果）

## 意图体系（PLM Domain Taxonomy）

系统基于 PLM 场景设计了 10 类核心意图：

| **Intent**        | **描述**    |
| ----------------- | --------- |
| ask_knowledge     | 查询知识 / 规则 |
| ask_howto         | 查询操作步骤    |
| query_status      | 查询状态      |
| summarize_content | 内容总结      |
| create_ticket     | 创建工单      |
| update_ticket     | 更新工单      |
| trigger_workflow  | 触发流程      |
| search_document   | 文档检索      |
| smalltalk         | 闲聊        |
| other             | 兜底        |

## 架构
```
                ┌──────────────────┐
                │   User Query     │
                └────────┬─────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Embedding Retrieval (语义召回) │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Classifier (意图分类模型)       │
        └────────────────┬───────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ Hybrid Policy (策略决策层)     │
        │ - 高风险意图对识别             │
        │ - 分数对比                     │
        │ - 澄清策略                     │
        └────────────────┬───────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Intent + Route   │
                └──────────────────┘
```
## 模型与数据结构
```
models/
├── classifier/
│   └── v3/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
│
├── embedding/
│   └── bge-small-zh-v1.5/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│
└── artifacts/
    ├── train_embeddings.npy
    ├── train_metadata.json
    ├── label2id.json
    └── id2label.json
```
## API 接口
### 意图识别
POST /intent/predict
```json
{
  "query": "帮我处理一下工单",
  "context": {
    "history": []
  },
  "options": {
    "debug": true
  }
}
```
### 就绪检查（推荐用于生产探针）
GET /ready
```json
{
  "status": "ready",
  "embedding_loaded": true,
  "classifier_loaded": true
}
```
### 示例
输入：
```
帮我处理一下工单
```
输出：
```json
{
  "top_intent": "update_ticket",
  "need_clarification": true,
  "clarification_question": "你是想新建一个单子，还是修改已有单子？"
}
```
Debug 输出
```json
{
  "debug": {
  "embedding_hits": [],
  "classifier_ranked_intents": []
}
}
```
## 启动方式
```shell
python -m uvicorn app.main:app --host 0.0.0.0 --port 48081 --reload
```

## 配置说明
```python
EMBEDDING_MODEL_DIR=models/embedding/bge-small-zh-v1.5
CLASSIFIER_MODEL_DIR=models/classifier/v3
ARTIFACTS_DIR=models/artifacts
```
支持通过环境变量覆盖。

## 线上效果指标
```text
Hybrid Accuracy: 0.869
Macro-F1:        0.856
Clarification:   9.8%
Top-2 Accuracy:  >0.93
```
 说明：
- 大部分请求可自动决策
- 小进入 clarification