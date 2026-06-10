"""
Benchmark 脚本：读取 CSV 中的 query，调用 NL2SQL 服务，输出 DSL 和 SQL。

用法:
  python scripts/benchmark.py \
      --input data/benchmark_queries.csv \
      --output reports/benchmark_result.csv \
      --url http://localhost:8089 \
      --user-key test \
      --concurrency 4
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def predict(session, url, user_key, query, struct_thred=0.7, timeout=60):
    try:
        r = session.post(
            f"{url}/nl2sql",
            params={"user_key": user_key},
            json={"query": query, "struct_thred": struct_thred, "execute": False, "trace": True},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        res = data.get("results", {})
        return {
            "query": query,
            "struct_desc": res.get("struct_desc", ""),
            "dsl": json.dumps(res.get("dsl") or {}, ensure_ascii=False),
            "sql": res.get("sql", ""),
            "similarity": res.get("similarity", ""),
            "error": data.get("message", "") if not res.get("sql") else "",
        }
    except Exception as e:
        return {"query": query, "struct_desc": "", "dsl": "", "sql": "", "similarity": "", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 CSV 文件，需包含 query 列")
    parser.add_argument("--output", default="reports/benchmark_result.csv")
    parser.add_argument("--url", default="http://localhost:8089")
    parser.add_argument("--user-key", default="test")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--struct-thred", type=float, default=0.7, help="reranker 阈值")
    parser.add_argument("--sample", type=int, default=None, help="随机采样数量，不指定则全量")
    args = parser.parse_args()

    # 读取 query
    queries = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("query", "").strip()
            if q:
                queries.append(q)

    if not queries:
        print("No queries found in input CSV")
        sys.exit(1)

    # 随机采样
    if args.sample and args.sample < len(queries):
        import random
        queries = random.sample(queries, args.sample)

    print(f"Loaded {len(queries)} queries from {args.input}")

    # 并发请求
    session = requests.Session()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(predict, session, args.url, args.user_key, q, args.struct_thred): q for q in queries}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            status = "OK" if r["sql"] else "FAIL"
            print(f"[{i}/{len(queries)}] {status} | {r['query'][:50]} | {r['struct_desc'][:60]}")

    # 按原始顺序排序
    query_order = {q: i for i, q in enumerate(queries)}
    results.sort(key=lambda x: query_order.get(x["query"], 0))

    # 写出
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "struct_desc", "dsl", "sql", "similarity", "error"])
        writer.writeheader()
        writer.writerows(results)

    success = sum(1 for r in results if r["sql"])
    print(f"\nDone: {success}/{len(results)} succeeded. Output: {args.output}")


if __name__ == "__main__":
    main()
