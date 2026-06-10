"""
DSL v2 → SQL 编译器
职责：纯确定性翻译，模型输出的简约 DSL → 可执行 SQL
"""

from typing import Any, Dict, List, Optional

import yaml
from pathlib import Path


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class CompilerV2:
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            config_dir = str(Path(__file__).parent / "configs")
        self.cfg = _load_yaml(f"{config_dir}/compiler_v2.yaml")
        self.field_map: Dict[str, str] = _load_yaml(f"{config_dir}/field_mapping.yaml")

        self._brand_map: Dict[str, str] = self.cfg.get("brand_map", {})
        self._cat_like: Dict[str, str] = self.cfg.get("category_like", {})
        self._global_filters: List[Dict] = self.cfg.get("global_filters", [])
        self._implicit_order: str = self.cfg.get("implicit_order", "")
        self._intent_select: Dict[str, List[str]] = self.cfg.get("intent_select", {})
        self._default_limit: int = self.cfg.get("default_limit", 5)
        self._max_limit: int = self.cfg.get("max_limit", 1000)
        self._force_like: set = set(self.cfg.get("force_like_cols", []))
        self._approx_ratio: float = self.cfg.get("approx_ratio", 0.1)
        self._table: str = self.cfg.get("table", "ads_mipd_aiplat_hiknow_product_insale_info_dd")

    # ---------- public ----------

    def compile(self, dsl: Dict) -> str:
        # intent=field 时校验 target_fields 是否为有效字段
        if dsl.get("intent") == "field":
            for tf in dsl.get("target_fields") or []:
                if tf not in self.field_map:
                    raise ValueError(f"无效的target字段: {tf}")

        select_cols = self._build_select(dsl)
        where_parts = self._build_where(dsl)
        order_parts = self._build_order(dsl)
        limit_val = self._build_limit(dsl)

        # 把 sort/filter 涉及的列也加入 SELECT（除 * 外）
        if select_cols != ["*"]:
            for col in self._extra_cols_from_filters(dsl):
                if col not in select_cols:
                    select_cols.append(col)
            sort_col = self._sort_col(dsl)
            if sort_col and sort_col not in select_cols:
                select_cols.append(sort_col)

        sel = ", ".join(select_cols) if select_cols != ["*"] else "*"
        sql = f"SELECT {sel} FROM {self._table}"
        if where_parts:
            sql += f" WHERE {' AND '.join(where_parts)}"
        if order_parts:
            sql += f" ORDER BY {', '.join(order_parts)}"
        sql += f" LIMIT {limit_val}"
        return sql

    # ---------- internal ----------

    def _col(self, field: str) -> str:
        return self.field_map.get(field, field)

    def _build_select(self, dsl: Dict) -> List[str]:
        intent = dsl.get("intent", "list")
        if intent == "spec":
            return ["*"]
        base = list(self._intent_select.get(intent, ["sale_model_name"]))
        if intent == "field":
            for f in dsl.get("target_fields") or []:
                c = self._col(f)
                if c not in base:
                    base.append(c)
        return base

    def _build_where(self, dsl: Dict) -> List[str]:
        parts: List[str] = []

        # brand
        brand = dsl.get("brand")
        if brand:
            eng = self._brand_map.get(brand, brand)
            parts.append(f"sale_brand_name = '{eng}'")

        # category
        cat = dsl.get("category")
        if cat:
            like_val = self._cat_like.get(cat, f"%{cat}%")
            parts.append(f"mid_class_name LIKE '{like_val}'")

        # filters
        for f in dsl.get("filters") or []:
            col = self._col(f["field"])
            op = f["op"]
            val = f["value"]

            if op == "~":
                # approximately
                try:
                    num = float(val)
                    lo = num * (1 - self._approx_ratio)
                    hi = num * (1 + self._approx_ratio)
                    parts.append(f"{col} BETWEEN {lo} AND {hi}")
                except ValueError:
                    parts.append(f"{col} = '{val}'")
            elif op == "!=" and val == "NULL":
                parts.append(f"{col} IS NOT NULL AND {col} != ''")
            elif col in self._force_like and op == "=":
                parts.append(f"{col} LIKE '%{val}%'")
            else:
                fv = self._format_val(val)
                sql_op = {"=": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "!=": "<>"}
                parts.append(f"{col} {sql_op.get(op, '=')} {fv}")

        # global implicit filters
        for g in self._global_filters:
            parts.append(f"{g['col']} {g['op']} '{g['value']}'")

        return parts

    def _build_order(self, dsl: Dict) -> List[str]:
        parts: List[str] = []
        sort = dsl.get("sort")
        if sort:
            col = self._col(sort["field"])
            direction = "DESC" if sort.get("dir") == "desc" else "ASC"
            parts.append(f"{col} {direction}")
        if self._implicit_order:
            parts.append(self._implicit_order)
        return parts

    def _build_limit(self, dsl: Dict) -> int:
        lim = dsl.get("limit")
        if lim is None or lim <= 0:
            return self._default_limit
        return min(int(lim), self._max_limit)

    def _sort_col(self, dsl: Dict) -> Optional[str]:
        sort = dsl.get("sort")
        return self._col(sort["field"]) if sort else None

    def _extra_cols_from_filters(self, dsl: Dict) -> List[str]:
        cols = []
        for f in dsl.get("filters") or []:
            c = self._col(f["field"])
            if c not in cols:
                cols.append(c)
        return cols

    def _format_val(self, val: str) -> str:
        try:
            float(val)
            return val
        except ValueError:
            return f"'{val}'"
