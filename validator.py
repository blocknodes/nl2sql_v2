"""
DSL v2 验证器
极简：JSON 合法 + field 白名单 + intent 合法 + 基本修复
"""

import json
from typing import Dict, Optional, Set


class ValidatorV2:
    def __init__(self, valid_fields: Set[str]):
        self.valid_fields = valid_fields
        self._valid_intents = {"list", "price", "spec", "field"}
        self._valid_ops = {"=", ">", "<", ">=", "<=", "!=", "~"}

    def parse_and_validate(self, raw: str, query: str) -> Dict:
        """
        解析 LLM 原始输出为 DSL dict，校验并修复。
        Raises ValueError 如果无法修复。
        """
        # 1. 提取 JSON（兼容 LLM 在前后加文字）
        dsl = self._extract_json(raw)

        # 2. intent 校验
        intent = dsl.get("intent", "list")
        if intent not in self._valid_intents:
            dsl["intent"] = "list"

        # 3. filters field 白名单
        filters = dsl.get("filters") or []
        valid_filters = []
        for f in filters:
            if not isinstance(f, dict):
                continue
            field = f.get("field", "")
            if field not in self.valid_fields:
                continue  # 丢弃无法识别的 field（防幻觉）
            op = f.get("op", "=")
            if op not in self._valid_ops:
                f["op"] = "="
            if "value" not in f:
                continue
            f["value"] = str(f["value"])
            valid_filters.append(f)
        dsl["filters"] = valid_filters

        # 4. sort 校验
        sort = dsl.get("sort")
        if sort and isinstance(sort, dict):
            if sort.get("field") not in self.valid_fields:
                dsl["sort"] = None
            elif sort.get("dir") not in ("asc", "desc"):
                sort["dir"] = "desc"

        # 5. limit
        lim = dsl.get("limit")
        if lim is not None:
            try:
                dsl["limit"] = int(lim)
            except (TypeError, ValueError):
                dsl["limit"] = None

        # 6. target_fields
        tfs = dsl.get("target_fields")
        if tfs and isinstance(tfs, list):
            dsl["target_fields"] = [f for f in tfs if f in self.valid_fields]
        else:
            dsl["target_fields"] = None

        # 7. category / brand 基本类型
        if dsl.get("category") not in (
            "电视", "空调", "冰箱", "冷柜", "洗衣机", "烘干机", "投影", "显示器", None
        ):
            dsl["category"] = None

        return dsl

    def _extract_json(self, raw: str) -> Dict:
        raw = raw.strip()
        # 尝试直接 parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # 尝试找第一个 { ... } 块
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 LLM 输出中提取合法 JSON: {raw[:200]}")
