"""
NL2SQL v2 Server (standalone)
启动：python server.py
"""

import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Query, Body
from pydantic import BaseModel

from pipeline import PipelineV2

app = FastAPI(title="NL2SQL Server v2", version="2.0")
_pipeline: Optional[PipelineV2] = None


def get_pipeline() -> PipelineV2:
    global _pipeline
    if _pipeline is None:
        model_server = os.getenv("MODEL_SERVER", "http://localhost:8088")
        print(f"[v2] 初始化 pipeline, MODEL_SERVER={model_server}")
        _pipeline = PipelineV2(model_server=model_server)
    return _pipeline


# ============== Reranker ==============

# DSL → 自然语言描述（用于 struct_desc）
_OP_DESC = {"=": "是", ">": "大于", "<": "小于", ">=": "大于等于", "<=": "小于等于", "!=": "不是", "~": "大约"}


def _build_field_desc_map():
    """从 prompt_builder 的字段字典构建 field -> 中文简短描述 映射。"""
    from prompt_builder import BASE_FIELDS, CATEGORY_FIELDS
    desc_map = {}
    for block in [BASE_FIELDS] + list(CATEGORY_FIELDS.values()):
        for line in block.strip().split("\n"):
            if ":" in line:
                field, desc = line.split(":", 1)
                field = field.strip()
                short = desc.strip().split("，")[0]
                desc_map[field] = short
    return desc_map


_field_desc_map: Optional[Dict] = None


def _get_field_desc_map():
    global _field_desc_map
    if _field_desc_map is None:
        _field_desc_map = _build_field_desc_map()
    return _field_desc_map


def dsl_to_natural_language(dsl: Dict) -> str:
    """将 v2 DSL 还原为自然语言描述。"""
    if not dsl:
        return ""
    desc_map = _get_field_desc_map()
    parts = []

    # 品牌
    if dsl.get("brand"):
        parts.append(f"品牌是{dsl['brand']}")

    # 类目
    if dsl.get("category"):
        parts.append(f"类目是{dsl['category']}")

    # filters
    for f in dsl.get("filters") or []:
        field = f.get("field", "")
        op = f.get("op", "=")
        value = f.get("value", "")
        field_desc = desc_map.get(field, field)
        op_desc = _OP_DESC.get(op, op)
        parts.append(f"{field_desc}{op_desc}{value}")

    # target_fields
    if dsl.get("target_fields"):
        target_descs = [desc_map.get(t, t) for t in dsl["target_fields"]]
        parts.append(f"查询{'、'.join(target_descs)}")
    elif dsl.get("intent") == "price":
        parts.append("查询价格")
    elif dsl.get("intent") == "spec":
        parts.append("查询所有参数")
    elif dsl.get("intent") == "list":
        parts.append("查询型号列表")

    # sort
    sort = dsl.get("sort")
    if sort:
        field_desc = desc_map.get(sort.get("field", ""), sort.get("field", ""))
        dir_desc = "从高到低" if sort.get("dir") == "desc" else "从低到高"
        parts.append(f"按{field_desc}{dir_desc}排序")

    # limit
    if dsl.get("limit"):
        parts.append(f"返回{dsl['limit']}条")

    return "，".join(parts)

_RERANK_PROMPT = """判断以下意图解析结果是否正确回答了用户的问题。评分 0-10 分，只输出数字。

用户问题：{query}
解析结果（自然语言描述）：{struct_desc}
解析结果（DSL）：{dsl}

评分标准：
- 10分：完全正确地理解了用户意图，条件、目标、排序都对
- 7-9分：基本正确，有小瑕疵
- 4-6分：部分正确，遗漏了关键条件或目标错误
- 1-3分：大部分错误
- 0分：完全无关

{few_shots}只输出一个 0-10 的整数："""


def _load_rerank_few_shots() -> list:
    """从配置文件加载 rerank few-shot 案例。"""
    import yaml
    path = Path(__file__).parent / "configs" / "rerank_few_shots.yaml"
    if path.exists():
        return yaml.safe_load(open(path, encoding="utf-8")) or []
    return []


def _format_rerank_few_shots(few_shots: list) -> str:
    if not few_shots:
        return ""
    lines = ["# 评分示例"]
    for ex in few_shots:
        lines.append(f"用户问题：{ex['query']}")
        lines.append(f"解析意图：{ex.get('struct_desc', '')}")
        lines.append(f"评分：{ex['score']}")
        lines.append("")
    return "\n".join(lines) + "\n"


_rerank_few_shots = _load_rerank_few_shots()


def rerank_score(pipeline: PipelineV2, query: str, sql: str, dsl: dict) -> tuple:
    """调用 LLM 对 DSL 打分，返回 (分数0~1, prompt, raw_output)。"""
    import json
    from pipeline import LLMClient

    if not dsl:
        return 0.0, "", ""

    struct_desc = dsl_to_natural_language(dsl)
    few_shots_str = _format_rerank_few_shots(_rerank_few_shots)
    prompt = _RERANK_PROMPT.format(
        query=query,
        struct_desc=struct_desc,
        dsl=json.dumps(dsl, ensure_ascii=False),
        few_shots=few_shots_str,
    )

    try:
        client = LLMClient(pipeline._model_server, pipeline._model_name)
        raw = client.generate(prompt).strip()
        client.close()
        # 提取数字
        import re
        m = re.search(r'\d+', raw)
        if m:
            return min(int(m.group()), 10) / 10.0, prompt, raw
    except Exception:
        pass
    return 0.5, prompt, ""  # 打分失败时给中间值，不拦截


class NL2SQLRequest(BaseModel):
    query: str
    struct_thred: Optional[float] = 0.5
    execute: Optional[bool] = None
    trace: Optional[bool] = False


def _execute_default() -> bool:
    return os.getenv("ENABLE_EXECUTE", "0").lower() in ("1", "true")


def _rerank_enabled() -> bool:
    return os.getenv("ENABLE_RERANK", "0").lower() in ("1", "true")


@app.get("/health")
def health():
    return {"status": "ok", "service": "NL2SQL v2"}


@app.post("/nl2sql")
def nl2sql(
    user_key: str = Query(...),
    request_body: NL2SQLRequest = Body(...),
) -> Dict[str, Any]:
    try:
        pipeline = get_pipeline()
        execute_flag = (
            request_body.execute
            if request_body.execute is not None
            else _execute_default()
        )
        result = pipeline.process(query=request_body.query, execute=execute_flag)

        # Reranker 打分：低于阈值则不输出 SQL
        rerank_threshold = request_body.struct_thred or 0.5
        score = 1.0
        rerank_prompt = ""
        rerank_raw = ""
        if _rerank_enabled() and result.get("sql"):
            score, rerank_prompt, rerank_raw = rerank_score(pipeline, request_body.query, result["sql"], result.get("dsl"))
            if score < rerank_threshold:
                result["sql"] = ""
                result["error"] = f"rerank_score={score:.2f} < threshold={rerank_threshold}"

        response = {
            "code": 200,
            "results": {
                "is_struct": bool(result["sql"]),
                "struct_results": {"raw_model_output": "是", "yes_probability": score},
                "sql": result["sql"],
                "subquery": "",
                "hit_filed": [],
                "llm_time": round(result.get("llm_time_ms", 0) / 1000, 3),
                "post_check_time": 0.0,
                "all_time": round(result.get("total_time_ms", 0) / 1000, 3),
                "table_name": "ads_mipd_aiplat_hiknow_product_insale_info_dd",
                "env": "dev",
                "struct_desc": dsl_to_natural_language(result.get("dsl")),
                "similarity": score,
            },
            "message": "success" if result["sql"] else result.get("error", "failed"),
        }

        if request_body.trace:
            response["results"]["dsl"] = result.get("dsl")
            response["results"]["latency_ms"] = result.get("total_time_ms", 0)
            response["results"]["executed"] = execute_flag
            response["results"]["data"] = result.get("data")
            response["results"]["exec_error"] = result.get("exec_error")
            response["results"]["out_of_domain"] = result.get("out_of_domain", False)
            response["trace"] = {
                "llm_prompt": (result.get("llm_prompt") or "").split("\n"),
                "llm_raw": (result.get("llm_raw") or "").split("\n"),
                "category_hint": result.get("category_hint"),
                "llm_time_ms": round(result.get("llm_time_ms", 0), 1),
                "total_time_ms": round(result.get("total_time_ms", 0), 1),
                "error": result.get("error", ""),
                "rerank_prompt": rerank_prompt,
                "rerank_raw": rerank_raw,
                "rerank_score": score,
            }

        return response

    except Exception as e:
        traceback.print_exc()
        return {
            "code": 200,
            "results": {
                "is_struct": False, "sql": "", "dsl": None,
                "llm_time": 0, "all_time": 0, "latency_ms": 0,
                "executed": False, "data": None, "exec_error": None,
            },
            "message": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)
