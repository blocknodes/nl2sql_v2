"""
NL2SQL v2 Server (standalone)
启动：python server.py
"""

import os
import traceback
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


class NL2SQLRequest(BaseModel):
    query: str
    struct_thred: Optional[float] = 0.5
    execute: Optional[bool] = None
    trace: Optional[bool] = False


def _execute_default() -> bool:
    return os.getenv("ENABLE_EXECUTE", "0").lower() in ("1", "true")


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

        response = {
            "code": 200,
            "results": {
                "is_struct": bool(result["sql"]),
                "struct_results": {"raw_model_output": "是", "yes_probability": 0.95},
                "sql": result["sql"],
                "dsl": result.get("dsl"),
                "subquery": "",
                "hit_filed": [],
                "llm_time": round(result.get("llm_time_ms", 0) / 1000, 3),
                "post_check_time": 0.0,
                "all_time": round(result.get("total_time_ms", 0) / 1000, 3),
                "table_name": "ads_mipd_aiplat_hiknow_product_insale_info_dd",
                "env": "dev",
                "struct_desc": "",
                "similarity": 1.0,
                "latency_ms": result.get("total_time_ms", 0),
                "executed": execute_flag,
                "data": result.get("data"),
                "exec_error": result.get("exec_error"),
                "out_of_domain": result.get("out_of_domain", False),
            },
            "message": "success" if result["sql"] else result.get("error", "failed"),
        }

        if request_body.trace:
            response["trace"] = {
                "llm_prompt": (result.get("llm_prompt") or "").split("\n"),
                "llm_raw": (result.get("llm_raw") or "").split("\n"),
                "category_hint": result.get("category_hint"),
                "llm_time_ms": round(result.get("llm_time_ms", 0), 1),
                "total_time_ms": round(result.get("total_time_ms", 0), 1),
                "error": result.get("error", ""),
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
