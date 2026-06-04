"""
NL2SQL v2 Pipeline (standalone)
串联：领域判断 → 类目检测 → prompt 构建(含 few-shot) → LLM 调用 → validator → compiler → SQL
"""

import json
import os
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import requests
import yaml

from compiler import CompilerV2
from prompt_builder import PromptBuilderV2, FewShotRetriever
from validator import ValidatorV2


class LLMClient:
    """简单的 vLLM chat completions 客户端"""

    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def generate(self, prompt: str) -> str:
        resp = self.session.post(
            self.url,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2048,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def close(self):
        self.session.close()


_DOMAIN_CHECK_PROMPT = """判断以下用户问题是否可以通过查询"家电产品信息数据库"来回答。

该数据库包含：电视、空调、冰箱、冷柜、洗衣机、烘干机、投影、显示器的产品信息，包括型号、价格、参数规格、品牌、颜色、尺寸、能效等级等。

只回答"是"或"否"。

用户问题：{query}"""


def _load_domain_blacklist(config_dir: str) -> List[str]:
    path = Path(config_dir) / "domain_blacklist.yaml"
    if path.exists():
        data = yaml.safe_load(open(path, encoding="utf-8")) or {}
        return data.get("keywords", [])
    return []


class PipelineV2:
    """全链路 v2 Pipeline"""

    MAX_RETRY = 1

    def __init__(self, config_dir: Optional[str] = None, model_server: Optional[str] = None):
        if config_dir is None:
            config_dir = str(Path(__file__).parent / "configs")
        self.config_dir = config_dir

        # 加载 field_mapping 建白名单
        fm = yaml.safe_load(open(f"{config_dir}/field_mapping.yaml", encoding="utf-8")) or {}
        valid_fields = set(fm.keys()) | {"salesPriceYuan", "salesModelName", "productPositionName"}

        self.compiler = CompilerV2(config_dir)
        self.retriever = FewShotRetriever(f"{config_dir}/few_shots.yaml")
        self.prompt_builder = PromptBuilderV2(self.retriever)
        self.validator = ValidatorV2(valid_fields)

        self._model_server = model_server or os.getenv("MODEL_SERVER", "http://localhost:8088")
        self._model_name = os.getenv("MODEL_NAME", "qwen4b")
        self._enable_domain_check = os.getenv("ENABLE_DOMAIN_CHECK", "1").lower() in ("1", "true")
        self._domain_blacklist = _load_domain_blacklist(config_dir)

        # Trace buffer
        self._traces: deque = deque(maxlen=10000)

    def _check_domain(self, query: str, llm_client: LLMClient) -> bool:
        """
        两级领域判断：
        1. 关键词黑名单命中 → 直接拒绝
        2. 黑名单未命中 → 调用 LLM 判断
        """
        for kw in self._domain_blacklist:
            if kw in query:
                return False
        raw = llm_client.generate(_DOMAIN_CHECK_PROMPT.format(query=query))
        return "是" in raw and "否" not in raw

    def process(self, query: str, execute: bool = False) -> Dict:
        """处理单条 query，返回结果 dict。"""
        t0 = time.time()

        # 关键词黑名单快速拦截（无需 LLM）
        if self._enable_domain_check:
            for kw in self._domain_blacklist:
                if kw in query:
                    total_ms = (time.time() - t0) * 1000
                    trace_id = str(uuid.uuid4())
                    self._traces.append({
                        "trace_id": trace_id, "query": query,
                        "error": "out_of_domain", "total_time_ms": round(total_ms, 1),
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    return {
                        "sql": "", "dsl": None, "llm_raw": "", "llm_prompt": "",
                        "llm_time_ms": total_ms, "total_time_ms": total_ms,
                        "error": "query不属于家电产品查询领域",
                        "trace_id": trace_id, "category_hint": None, "out_of_domain": True,
                    }

        # 类目检测 + prompt 构建
        category_hint = self._detect_category(query)
        prompt = self.prompt_builder.build(query, category_hint)

        # 并发调用：domain check + NL2SQL 同时发出
        from concurrent.futures import ThreadPoolExecutor, Future

        llm_client = LLMClient(self._model_server, self._model_name)
        domain_future: Optional[Future] = None

        if self._enable_domain_check:
            pool = ThreadPoolExecutor(max_workers=2)
            domain_future = pool.submit(
                llm_client.generate, _DOMAIN_CHECK_PROMPT.format(query=query)
            )
            nl2sql_future = pool.submit(llm_client.generate, prompt)
            pool.shutdown(wait=False)
        else:
            nl2sql_future = None

        # LLM 调用 + 重试
        dsl = None
        llm_raw = ""
        error = ""
        llm_time_ms = 0.0

        try:
            for attempt in range(self.MAX_RETRY + 1):
                lt0 = time.time()
                if attempt == 0 and nl2sql_future is not None:
                    llm_raw = nl2sql_future.result(timeout=60)
                else:
                    llm_raw = llm_client.generate(prompt)
                llm_time_ms = (time.time() - lt0) * 1000
                try:
                    dsl = self.validator.parse_and_validate(llm_raw, query)
                    break
                except ValueError as e:
                    error = str(e)
                    if attempt < self.MAX_RETRY:
                        prompt += f"\n\n上次输出有误：{e}。请重新输出纯JSON。"
                        error = ""
        except Exception as e:
            error = f"LLM调用失败: {e}"
        finally:
            llm_client.close()

        # 检查 domain check 结果（与 NL2SQL 并发完成）
        if domain_future is not None:
            try:
                domain_raw = domain_future.result(timeout=60)
                in_domain = "是" in domain_raw and "否" not in domain_raw
            except Exception:
                in_domain = True
            if not in_domain:
                total_ms = (time.time() - t0) * 1000
                trace_id = str(uuid.uuid4())
                self._traces.append({
                    "trace_id": trace_id, "query": query,
                    "error": "out_of_domain", "total_time_ms": round(total_ms, 1),
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return {
                    "sql": "", "dsl": None, "llm_raw": "", "llm_prompt": "",
                    "llm_time_ms": total_ms, "total_time_ms": total_ms,
                    "error": "query不属于家电产品查询领域",
                    "trace_id": trace_id, "category_hint": None, "out_of_domain": True,
                }

        if error or dsl is None:
            total_ms = (time.time() - t0) * 1000
            trace_id = str(uuid.uuid4())
            self._traces.append({
                "trace_id": trace_id, "query": query, "category_hint": category_hint,
                "llm_raw": llm_raw, "dsl": dsl, "sql": "",
                "llm_time_ms": round(llm_time_ms, 1),
                "total_time_ms": round(total_ms, 1), "error": error,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return {
                "sql": "", "dsl": dsl, "llm_raw": llm_raw, "llm_prompt": prompt,
                "llm_time_ms": llm_time_ms, "total_time_ms": total_ms,
                "error": error, "trace_id": trace_id, "category_hint": category_hint,
            }

        # 编译
        sql = self.compiler.compile(dsl)

        total_ms = (time.time() - t0) * 1000
        result = {
            "sql": sql, "dsl": dsl, "llm_raw": llm_raw, "llm_prompt": prompt,
            "llm_time_ms": llm_time_ms, "total_time_ms": total_ms, "error": "",
        }

        # 执行模式
        if execute:
            try:
                from executor import MySQLExecutor
                ex = MySQLExecutor()
                result["data"] = ex.execute(sql)
                result["exec_error"] = None
            except Exception as e:
                result["data"] = None
                result["exec_error"] = str(e)

        # Trace
        trace_id = str(uuid.uuid4())
        result["trace_id"] = trace_id
        result["category_hint"] = category_hint
        self._traces.append({
            "trace_id": trace_id, "query": query, "category_hint": category_hint,
            "dsl": dsl, "sql": sql,
            "llm_time_ms": round(llm_time_ms, 1),
            "total_time_ms": round(total_ms, 1), "error": "",
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        return result

    def _detect_category(self, query: str) -> Optional[str]:
        """关键词匹配类目"""
        kw_map = {
            "电视": ["电视", "TV"],
            "空调": ["空调"],
            "冰箱": ["冰箱"],
            "冷柜": ["冷柜", "冰柜"],
            "洗衣机": ["洗衣机"],
            "烘干机": ["烘干机", "干衣机"],
            "投影": ["投影", "投影仪"],
            "显示器": ["显示器"],
        }
        for cat, kws in kw_map.items():
            for kw in kws:
                if kw in query:
                    return cat
        return None
