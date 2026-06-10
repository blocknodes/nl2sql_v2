# NL2SQL v2

基于意图提取 + 确定性编译的自然语言转 SQL 服务。

## 架构

```
用户 query
    ↓
[型号提取] ─ 正则匹配 → __MODEL__ 占位符（过滤参数值如4GB/120Hz）
    ↓
[领域判断] ─ 白名单正向最大匹配（configs/domain_whitelist.yaml）─→ 拒绝
    ↓ 通过
[类目检测] ─ 关键词匹配
    ↓
[Prompt构建] ─ 按类目裁剪字段 + BM25 动态 few-shot + 占位符规则
    ↓
[LLM调用] ─ Qwen3-4B → 输出 JSON DSL
    ↓
[Validator] ─ JSON提取 + field白名单 + intent修正 + 兜底修复
    ↓
[型号还原] ─ __MODEL__ → 真实值 + 自动判断字段(salesModelName/promotionName)
    ↓
[Compiler] ─ DSL → SQL 确定性翻译
    ↓
[Reranker] ─ (可选) LLM 对 DSL 打分，低于阈值拒绝输出
    ↓
SQL + struct_desc（自然语言描述）
```

## 快速启动

```bash
pip install -r requirements.txt

export MODEL_SERVER=http://localhost:8090
export ENABLE_DOMAIN_CHECK=1    # 白名单领域判断（默认开）
export ENABLE_RERANK=0          # Reranker 开关（默认关）
export ENABLE_EXECUTE=0         # SQL 执行开关

python server.py
# 监听 0.0.0.0:8089
```

## 接口

```bash
# 基础调用
curl -X POST 'http://localhost:8089/nl2sql?user_key=test' \
  -H "Content-Type: application/json" \
  -d '{"query":"维迪亚100英寸以上最贵的电视","struct_thred":0.7,"execute":false,"trace":false}'

# trace=true 输出完整 debug 信息（dsl/prompt/rerank等）
curl -X POST 'http://localhost:8089/nl2sql?user_key=test' \
  -H "Content-Type: application/json" \
  -d '{"query":"U7S电视分辨率","trace":true}'
```

**响应字段（默认）：**
- `sql` — 生成的 SQL
- `struct_desc` — DSL 的自然语言描述
- `similarity` — reranker 分数（未启用时为 1.0）
- `is_struct` — 是否成功结构化

**trace=true 额外输出：** `dsl`、`llm_prompt`、`llm_raw`、`rerank_prompt`、`rerank_raw` 等

## 评测

```bash
# API 模式评测
python scripts/eval.py \
    --input data/eval_real_queries_v2.jsonl \
    --mode api --url http://localhost:8089 \
    --user-key test --concurrency 8

# Benchmark（CSV 输入，支持采样）
python scripts/benchmark.py \
    --input data/queries.csv \
    --output reports/benchmark_result.csv \
    --sample 50 --struct-thred 0.7
```

## GRPO 训练

```bash
pip install trl transformers datasets torch accelerate flash-attn

python scripts/train_grpo.py \
    --model_name_or_path Qwen/Qwen3-4B \
    --dataset data/train_v2.jsonl \
    --output_dir output/grpo_nl2sql \
    --num_train_epochs 3 \
    --num_generations 4
```

**Reward 函数（4 个递进）：**
1. `format` — JSON 可解析
2. `field` — 字段在白名单中的比例
3. `dsl_match` — 与 GT DSL 细粒度匹配
4. `sql` — 编译后 SQL 与 GT SQL 一致

训练 rollout 日志输出到 `output/grpo_nl2sql/rollout_log.jsonl`。

## 目录结构

```
nl2sql_v2/
├── configs/
│   ├── compiler_v2.yaml         # 编译器规则(品牌映射/全局filter/force_like等)
│   ├── field_mapping.yaml       # 逻辑字段 → 物理列
│   ├── few_shots.yaml           # NL2SQL BM25 检索案例库
│   ├── rerank_few_shots.yaml    # Reranker 打分 few-shot
│   ├── domain_whitelist.yaml    # 领域白名单词库
│   ├── domain_blacklist.yaml    # 领域黑名单(已弃用)
│   └── dsl_v2.schema.json
├── data/                        # 评测/训练数据
├── reports/                     # 评测输出
├── scripts/
│   ├── eval.py                  # 精确匹配评测
│   ├── benchmark.py             # 批量 benchmark（CSV）
│   ├── gen_train_data.py        # 训练数据生成
│   └── train_grpo.py            # GRPO 强化学习训练
├── pipeline.py                  # 全链路 Pipeline
├── prompt_builder.py            # Prompt + BM25 + 型号提取/还原
├── validator.py                 # DSL 校验修复
├── compiler.py                  # DSL → SQL
├── executor.py                  # MySQL 只读执行器
├── server.py                    # FastAPI 入口 + Reranker
├── requirements.txt
└── README.md
```

## Badcase 修复

| 优先级 | 手段 | 说明 |
|--------|------|------|
| P0 | `configs/few_shots.yaml` 加案例 | BM25 召回，重启生效 |
| P0 | `configs/domain_whitelist.yaml` 加词 | 修复域外误判 |
| P1 | `configs/rerank_few_shots.yaml` 加案例 | 修复 reranker 误拦 |
| P2 | 修改 prompt 规则/字段字典 (`prompt_builder.py`) | 重启生效 |
| P3 | `compiler_v2.yaml` 加规则 | 重启生效 |
| P4 | `validator.py` 加修正逻辑 | 确定性兜底 |
| P5 | GRPO 训练 | 天级 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_SERVER` | `http://localhost:8088` | LLM 服务地址 |
| `MODEL_NAME` | `qwen4b` | 模型名 |
| `ENABLE_DOMAIN_CHECK` | `1` | 白名单领域判断 |
| `ENABLE_RERANK` | `0` | Reranker 打分拦截 |
| `ENABLE_EXECUTE` | `0` | 是否执行 SQL |
