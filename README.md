# NL2SQL v2

基于意图提取 + 确定性编译的自然语言转 SQL 服务。

## 架构

```
用户 query
    ↓
[领域判断] ─ 黑名单关键词 + LLM 是否属于家电产品查询 ─→ 拒绝
    ↓ 通过
[类目检测] ─ 关键词匹配
    ↓
[Prompt构建] ─ 按类目裁剪字段 + BM25 动态 few-shot
    ↓
[LLM调用] ─ Qwen3-4B → 输出 JSON DSL
    ↓
[Validator] ─ JSON提取 + field白名单 + 兜底修复
    ↓
[Compiler] ─ DSL → SQL 确定性翻译
    ↓
SQL + (可选)执行结果
```

## 快速启动

```bash
pip install -r requirements.txt

export MODEL_SERVER=http://localhost:8090
export ENABLE_DOMAIN_CHECK=1    # 领域判断开关（默认开）
export ENABLE_EXECUTE=0         # SQL 执行开关

python server.py
# 监听 0.0.0.0:8089
```

## 接口

```bash
curl -X POST 'http://localhost:8089/nl2sql?user_key=test' \
  -H "Content-Type: application/json" \
  -d '{"query":"维迪亚100英寸以上最贵的电视","execute":false,"trace":true}'
```

## 评测

```bash
python scripts/eval.py \
    --input data/eval_real_queries_v2.jsonl \
    --mode api --url http://localhost:8089 \
    --user-key test --concurrency 8 \
    --output reports/eval_result.jsonl
```

## 目录结构

```
nl2sql_v2/
├── configs/
│   ├── compiler_v2.yaml         # 编译器规则(品牌映射/全局filter/force_like等)
│   ├── field_mapping.yaml       # 逻辑字段 → 物理列
│   ├── few_shots.yaml           # BM25 检索案例库
│   ├── domain_blacklist.yaml    # 领域黑名单关键词(可配置)
│   └── dsl_v2.schema.json
├── data/                        # 评测数据
├── reports/                     # 评测输出
├── scripts/
│   └── eval.py                  # 评测脚本
├── pipeline.py                  # 全链路 Pipeline
├── prompt_builder.py            # Prompt + BM25 Retriever
├── validator.py                 # DSL 校验修复
├── compiler.py                  # DSL → SQL
├── executor.py                  # MySQL 只读执行器
├── server.py                    # FastAPI 入口
├── requirements.txt
└── README.md
```

## Badcase 修复

| 优先级 | 手段 | 说明 |
|--------|------|------|
| P0 | `configs/few_shots.yaml` 加案例 | BM25 召回，秒级生效 |
| P1 | `configs/domain_blacklist.yaml` 加关键词 | 拦截非领域 query |
| P2 | 修改 prompt 规则/字段字典 | 重启生效 |
| P3 | `compiler_v2.yaml` 加规则 | 重启生效 |
| P4 | 模型 SFT 重训 | 天级 |
