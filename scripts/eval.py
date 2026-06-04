"""
NL2SQL v2 评测脚本

用法：
  # API 模式
  python scripts/eval.py --input data/eval_real_queries_v2.jsonl --mode api \
      --url http://localhost:8089 --user-key xxx --concurrency 8

  # 本地模式（需 vLLM 可达）
  python scripts/eval.py --input data/eval_200_v2.jsonl --mode local

  # 只跑前 N 条
  python scripts/eval.py --input data/eval_200_v2.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 把项目根目录加到 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import requests


# ============== SQL 规范化 ==============

_WS_RE = re.compile(r"\s+")
_SELECT_RE = re.compile(r"^\s*SELECT\s+(.+?)\s+FROM\s+", re.IGNORECASE | re.DOTALL)
_WHERE_RE = re.compile(r"\bWHERE\s+(.+?)(?=\s+(?:GROUP|ORDER|LIMIT|HAVING)\b|\s*$)", re.IGNORECASE | re.DOTALL)


def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    s = _WS_RE.sub(" ", sql.strip())
    m = _SELECT_RE.match(s)
    if m:
        cols = sorted(c.strip() for c in m.group(1).split(","))
        s = s[:m.start(1)] + ", ".join(cols) + s[m.end(1):]
    m2 = _WHERE_RE.search(s)
    if m2 and " OR " not in m2.group(1).upper():
        conds = sorted(c.strip() for c in re.split(r"\s+AND\s+", m2.group(1), flags=re.IGNORECASE))
        s = s[:m2.start(1)] + " AND ".join(conds) + s[m2.end(1):]
    return s


# ============== DSL 比较 ==============

def _filter_key(f: Dict) -> Tuple:
    return (str(f.get("field")), str(f.get("op")), str(f.get("value")).strip("%"))


def cond_match(pred: Dict, gt: Dict) -> bool:
    if (pred.get("category") or None) != (gt.get("category") or None):
        return False
    if (pred.get("brand") or None) != (gt.get("brand") or None):
        return False
    a = sorted([_filter_key(f) for f in pred.get("filters") or []])
    b = sorted([_filter_key(f) for f in gt.get("filters") or []])
    return a == b


def targets_match(pred: Dict, gt: Dict) -> bool:
    if (pred.get("intent") or "list") != (gt.get("intent") or "list"):
        return False
    a = sorted(pred.get("target_fields") or [])
    b = sorted(gt.get("target_fields") or [])
    return a == b


def order_match(pred: Dict, gt: Dict) -> bool:
    ps = pred.get("sort") or {}
    gs = gt.get("sort") or {}
    return (ps.get("field"), ps.get("dir")) == (gs.get("field"), gs.get("dir"))


def count_match(pred: Dict, gt: Dict) -> bool:
    return (pred.get("limit") or None) == (gt.get("limit") or None)


def dsl_match(pred: Dict, gt: Dict) -> bool:
    return cond_match(pred, gt) and targets_match(pred, gt) and order_match(pred, gt) and count_match(pred, gt)


# ============== 后端 ==============

class Backend:
    def predict(self, query: str) -> Tuple[Dict, str, str]:
        raise NotImplementedError


class APIBackend(Backend):
    def __init__(self, url: str, user_key: str, timeout: int = 60):
        self.url = url.rstrip("/")
        self.user_key = user_key
        self.timeout = timeout
        self.session = requests.Session()

    def predict(self, query: str) -> Tuple[Dict, str, str]:
        try:
            r = self.session.post(
                f"{self.url}/nl2sql",
                params={"user_key": self.user_key},
                json={"query": query, "struct_thred": 0.5, "execute": False},
                timeout=self.timeout,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            return {}, "", f"http_error: {e}"

        results = payload.get("results") or {}
        sql = results.get("sql") or ""
        dsl = results.get("dsl") or {}
        if not sql:
            return dsl, "", "empty_sql"
        return dsl, sql, ""


class LocalBackend(Backend):
    def __init__(self, config_dir: Optional[str] = None, model_server: Optional[str] = None):
        from pipeline import PipelineV2
        self.pipeline = PipelineV2(config_dir=config_dir, model_server=model_server)

    def predict(self, query: str) -> Tuple[Dict, str, str]:
        try:
            r = self.pipeline.process(query, execute=False)
        except Exception as e:
            return {}, "", f"pipeline_error: {e}"
        return r.get("dsl") or {}, r.get("sql") or "", ""


# ============== Compiler（gt DSL → 金 SQL）==============

def build_gt_compiler(config_dir: Optional[str] = None):
    from compiler import CompilerV2
    if config_dir is None:
        config_dir = str(Path(__file__).parent.parent / "configs")
    return CompilerV2(config_dir)


def gt_to_sql(compiler, gt_dsl: Dict) -> str:
    try:
        return compiler.compile(json.loads(json.dumps(gt_dsl)))
    except Exception as e:
        return f"<compile_error: {e}>"


# ============== 评测主循环 ==============

@dataclass
class CaseResult:
    id: int
    category: str
    query: str
    pred_dsl: Dict
    pred_sql: str
    gt_dsl: Dict
    gt_sql: str
    error: str = ""
    elapsed_ms: float = 0.0
    sql_match: int = 0
    dsl_match: int = 0
    cond_match: int = 0
    targets_match: int = 0
    order_match: int = 0
    count_match: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "category": self.category, "query": self.query,
            "metrics": {
                "sql_match": self.sql_match, "dsl_match": self.dsl_match,
                "cond_match": self.cond_match, "targets_match": self.targets_match,
                "order_match": self.order_match, "count_match": self.count_match,
            },
            "elapsed_ms": round(self.elapsed_ms, 1), "error": self.error,
            "pred": {"dsl": self.pred_dsl, "sql": self.pred_sql},
            "gt": {"dsl": self.gt_dsl, "sql": self.gt_sql},
        }


def evaluate_case(case: Dict, backend: Backend, compiler) -> CaseResult:
    gt_dsl = case["dsl"]
    gt_sql = gt_to_sql(compiler, gt_dsl)
    category = gt_dsl.get("category") or ""

    t0 = time.time()
    pred_dsl, pred_sql, err = backend.predict(case["query"])
    elapsed_ms = (time.time() - t0) * 1000

    res = CaseResult(
        id=case["id"], category=category, query=case["query"],
        pred_dsl=pred_dsl, pred_sql=pred_sql,
        gt_dsl=gt_dsl, gt_sql=gt_sql,
        error=err, elapsed_ms=elapsed_ms,
    )
    if err:
        return res

    res.sql_match = int(normalize_sql(pred_sql) == normalize_sql(gt_sql))
    res.cond_match = int(cond_match(pred_dsl, gt_dsl))
    res.targets_match = int(targets_match(pred_dsl, gt_dsl))
    res.order_match = int(order_match(pred_dsl, gt_dsl))
    res.count_match = int(count_match(pred_dsl, gt_dsl))
    res.dsl_match = int(res.cond_match and res.targets_match and res.order_match and res.count_match)
    return res


# ============== 报表 ==============

def _avg(arr: List[float]) -> float:
    return round(sum(arr) / len(arr), 4) if arr else 0.0


def print_report(results: List[CaseResult]) -> None:
    cats: Dict[str, List[CaseResult]] = {}
    for r in results:
        cats.setdefault(r.category or "未知", []).append(r)

    def row(label: str, rs: List[CaseResult]) -> List:
        n = len(rs)
        errs = sum(1 for r in rs if r.error)
        return [label, n, errs,
                _avg([r.sql_match for r in rs]), _avg([r.dsl_match for r in rs]),
                _avg([r.cond_match for r in rs]), _avg([r.targets_match for r in rs]),
                _avg([r.order_match for r in rs]), _avg([r.count_match for r in rs]),
                round(_avg([r.elapsed_ms for r in rs]), 1)]

    headers = ["category", "n", "err", "sql", "dsl", "cond", "tgt", "order", "cnt", "avg_ms"]
    rows = [row("OVERALL", results)] + [row(c, rs) for c, rs in sorted(cats.items())]

    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print()
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))

    o = rows[0]
    print(f"\n[summary] n={o[1]} sql={o[3]} dsl={o[4]} cond={o[5]} tgt={o[6]} order={o[7]} cnt={o[8]}")


def _progress_log(r: CaseResult) -> None:
    flag = "OK" if (r.sql_match or r.dsl_match) and not r.error else ("ER" if r.error else "FA")
    print(f"  [{flag}] #{r.id:03d} [{r.category}] sql={r.sql_match} dsl={r.dsl_match} "
          f"({r.elapsed_ms:.0f}ms) {r.query[:40]}")
    if flag != "OK":
        print(f"        pred_dsl: {json.dumps(r.pred_dsl, ensure_ascii=False)}")
        print(f"        gt_dsl:   {json.dumps(r.gt_dsl, ensure_ascii=False)}")


# ============== 主入口 ==============

def main() -> int:
    p = argparse.ArgumentParser(description="NL2SQL v2 evaluator")
    p.add_argument("--input", default="data/eval_real_queries_v2.jsonl")
    p.add_argument("--mode", choices=["api", "local"], default="api")
    p.add_argument("--url", default="http://localhost:8089")
    p.add_argument("--user-key", default="test")
    p.add_argument("--config-dir", default=None)
    p.add_argument("--model-server", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--output", default="reports/eval_result.jsonl")
    args = p.parse_args()

    # 读输入
    cases: List[Dict] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"[eval] loaded {len(cases)} cases from {args.input}")

    # 后端
    if args.mode == "api":
        backend: Backend = APIBackend(args.url, args.user_key)
    else:
        backend = LocalBackend(config_dir=args.config_dir, model_server=args.model_server)

    compiler = build_gt_compiler(args.config_dir)

    # 评测
    results: List[CaseResult] = []
    t_start = time.time()

    if args.mode == "api" and args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(evaluate_case, c, backend, compiler): c for c in cases}
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                _progress_log(r)
    else:
        for c in cases:
            r = evaluate_case(c, backend, compiler)
            results.append(r)
            _progress_log(r)

    total_s = time.time() - t_start
    results.sort(key=lambda x: x.id)

    print_report(results)
    print(f"[eval] total wall: {total_s:.2f}s ({total_s*1000/max(1,len(results)):.1f} ms/case avg)")

    # 写详细结果
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"[eval] results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
