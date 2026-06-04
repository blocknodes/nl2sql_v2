"""
MySQL 执行器
职责：执行 Layer 3 编译出的只读 SELECT，返回结构化数据。

安全约束：
1. 仅放行 SELECT（拒绝多语句、非 SELECT、注释中的 DDL/DML）
2. 强制 LIMIT 上限（compiler 已加 LIMIT，这里再叠加一层兜底）
3. 单次查询超时（read_timeout / connect_timeout）
4. 连接每次请求新建并 close（小流量场景）

环境变量：
- MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB
- MYSQL_HARD_LIMIT 兜底行数上限（默认 1000）
- MYSQL_TIMEOUT 单次查询秒级超时（默认 10）
"""

import os
import re
import time
from datetime import date, datetime, time as dtime
from decimal import Decimal
from typing import Any, Dict, List, Optional

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError as e:  # pragma: no cover
    pymysql = None  # 延迟报错：仅当真正需要执行时才抛
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


# 仅允许以 SELECT 开头（忽略前导空白和 SQL 注释）
_SELECT_ONLY_RE = re.compile(
    r"""^\s*                  # leading whitespace
        (?:/\*.*?\*/\s*)*     # /* ... */ block comments
        (?:--[^\n]*\n\s*)*    # -- line comments
        SELECT\b              # must start with SELECT
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

# 危险关键词（即便混在 SELECT 之后也禁止）—— 编译器不会产生，但作为防御
_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|RENAME|REPLACE|CALL|HANDLER|LOAD\s+DATA|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b",
    re.IGNORECASE,
)


class ExecutorConfig:
    """从环境变量读取连接配置。延迟读取，便于测试覆盖。"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        hard_limit: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.host = host or os.getenv("MYSQL_HOST", "10.19.37.217")
        self.port = int(port if port is not None else os.getenv("MYSQL_PORT", "9030"))
        self.user = user or os.getenv("MYSQL_USER", "ds_hiknow_ro")
        self.password = password if password is not None else os.getenv("MYSQL_PASSWORD", "dev_x5gN")
        self.database = database or os.getenv("MYSQL_DB", "ads")
        self.hard_limit = int(hard_limit if hard_limit is not None else os.getenv("MYSQL_HARD_LIMIT", "1000"))
        self.timeout = int(timeout if timeout is not None else os.getenv("MYSQL_TIMEOUT", "10"))

    def masked(self) -> Dict[str, Any]:
        """供日志/调试输出，隐去密码"""
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "database": self.database,
            "hard_limit": self.hard_limit,
            "timeout": self.timeout,
        }


def _to_jsonable(value: Any) -> Any:
    """把 MySQL 返回的特殊类型转为 JSON 可序列化对象"""
    if value is None:
        return None
    if isinstance(value, Decimal):
        # 优先保留为字符串，避免精度损失；若用户希望浮点可在外层转
        return str(value)
    if isinstance(value, (datetime, date, dtime)):
        return value.isoformat(sep=" ")
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    return value


class SQLValidationError(ValueError):
    """SQL 安全校验失败"""


class MySQLExecutor:
    """只读 SQL 执行器"""

    def __init__(self, config: Optional[ExecutorConfig] = None):
        if pymysql is None:  # pragma: no cover
            raise RuntimeError(
                f"pymysql 未安装，无法使用执行模式: {_IMPORT_ERR}. 请 pip install pymysql"
            )
        self.config = config or ExecutorConfig()

    # ---------- public ----------

    def execute(self, sql: str) -> Dict[str, Any]:
        """
        执行只读 SELECT 并返回结构化数据。

        Returns:
            {
                "columns": ["c1", "c2", ...],
                "rows": [[v1, v2, ...], ...],
                "row_count": N,
                "truncated": bool,
                "exec_time_ms": float,
                "executed_sql": "SELECT ..."
            }

        Raises:
            SQLValidationError: SQL 不是 SELECT 或包含禁用语句
            pymysql.MySQLError: 连接 / 执行错误
        """
        safe_sql = self._validate_and_cap(sql)

        t0 = time.time()
        conn = pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            charset="utf8mb4",
            connect_timeout=self.config.timeout,
            read_timeout=self.config.timeout,
            write_timeout=self.config.timeout,
            cursorclass=DictCursor,
            autocommit=True,
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(safe_sql)
                raw_rows = cursor.fetchall() or []
                # cursor.description: 顺序保留
                columns = [d[0] for d in (cursor.description or [])]
        finally:
            try:
                conn.close()
            except Exception:
                pass

        rows: List[List[Any]] = []
        for r in raw_rows:
            rows.append([_to_jsonable(r.get(c)) for c in columns])

        elapsed_ms = (time.time() - t0) * 1000
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": len(rows) >= self.config.hard_limit,
            "exec_time_ms": round(elapsed_ms, 1),
            "executed_sql": safe_sql,
        }

    # ---------- internal ----------

    def _validate_and_cap(self, sql: str) -> str:
        """SQL 安全校验 + LIMIT 兜底"""
        if not sql or not sql.strip():
            raise SQLValidationError("SQL 为空")

        # 拒绝多语句：strip 末尾分号后不能再有第二条
        stripped = sql.strip().rstrip(";").strip()
        if ";" in stripped:
            raise SQLValidationError("禁止多语句执行")

        if not _SELECT_ONLY_RE.match(stripped):
            raise SQLValidationError("仅允许 SELECT 查询")

        if _FORBIDDEN_RE.search(stripped):
            raise SQLValidationError("SQL 包含禁用关键词")

        # LIMIT 兜底（compiler 已加 LIMIT，但仍校验一次）
        if not re.search(r"\bLIMIT\s+\d+", stripped, re.IGNORECASE):
            stripped = f"{stripped} LIMIT {self.config.hard_limit}"
        else:
            # 把现有 LIMIT 截到 hard_limit 以内
            def _cap(m: re.Match) -> str:
                n = int(m.group(1))
                capped = min(n, self.config.hard_limit)
                return f"LIMIT {capped}"

            stripped = re.sub(r"\bLIMIT\s+(\d+)", _cap, stripped, flags=re.IGNORECASE)

        return stripped
