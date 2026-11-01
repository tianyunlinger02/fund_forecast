"""Small SQLite repository with immutable source versions and transactional paper state."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date
from decimal import Decimal, DecimalException
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit

from .advice_evidence import (
    MAX_CAPTURE_BYTES,
    MAX_EVIDENCE_LINKS,
    MAX_EVIDENCE_READ_BYTES,
    MAX_EVIDENCE_RECORDS,
    MAX_RECORD_BYTES,
    bounded_json,
    normalize_capture,
    normalize_record,
    research_thesis_content,
)
from .domain import Fact, Product, Profile, ValidationError, canonical_json, digest, timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS products(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles(
 id TEXT PRIMARY KEY, product_id TEXT NOT NULL, effective_from TEXT NOT NULL,
 effective_to TEXT, observed_at TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots(
 id TEXT PRIMARY KEY, source TEXT NOT NULL, product_id TEXT NOT NULL,
 observed_at TEXT NOT NULL, source_url TEXT NOT NULL, status TEXT NOT NULL,
 warnings TEXT NOT NULL, content_hash TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts(
 id TEXT PRIMARY KEY, product_id TEXT NOT NULL, kind TEXT NOT NULL,
 event_date TEXT NOT NULL, observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
 payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS fact_lookup ON facts(product_id,kind,event_date);
CREATE INDEX IF NOT EXISTS snapshot_product_time ON snapshots(product_id,observed_at);
CREATE TABLE IF NOT EXISTS accounts(
 id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL,
 initial_cash TEXT NOT NULL,mode TEXT NOT NULL,last_advanced_at TEXT NOT NULL,
 payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders(
 id TEXT PRIMARY KEY,account_id TEXT NOT NULL,product_id TEXT NOT NULL,
 side TEXT NOT NULL,state TEXT NOT NULL,submitted_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_events(
 id INTEGER PRIMARY KEY AUTOINCREMENT,event_key TEXT UNIQUE NOT NULL,
 account_id TEXT NOT NULL,order_id TEXT,product_id TEXT,event_type TEXT NOT NULL,
 economic_date TEXT NOT NULL,recorded_at TEXT NOT NULL,payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_account ON ledger_events(account_id,id);
CREATE TABLE IF NOT EXISTS journal(
 id TEXT PRIMARY KEY,product_id TEXT,at TEXT NOT NULL,decision TEXT NOT NULL,
 reason TEXT NOT NULL,source_url TEXT NOT NULL,evidence TEXT NOT NULL
);
"""


# Historical checker tables remain readable when present, but no live operation
# creates or writes them. Raw public-document evidence has its own explicit init.
EVIDENCE_TABLES = (
    "advice_evidence_captures", "advice_evidence_records", "advice_evidence_links",
)
FUND_REVIEW_TABLES = (*EVIDENCE_TABLES, "daily_advice_checks", "advice_policy_activations")
EVIDENCE_SCHEMA = """
CREATE TABLE advice_evidence_captures(
 id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
 family TEXT NOT NULL, source TEXT NOT NULL, upstream TEXT NOT NULL, source_url TEXT NOT NULL,
 observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('ok','reviewed','error')),
 raw_sha256 TEXT NOT NULL, raw_bytes BLOB NOT NULL CHECK(length(raw_bytes)<=8388608),
 payload TEXT NOT NULL CHECK(length(CAST(payload AS BLOB))<=262144)
);
CREATE INDEX advice_capture_lookup ON advice_evidence_captures(product_id,observed_at);
CREATE TABLE advice_evidence_records(
 id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
 family TEXT NOT NULL, source TEXT NOT NULL, upstream TEXT NOT NULL, source_url TEXT NOT NULL,
 report_date TEXT NOT NULL, observed_at TEXT NOT NULL, available_at TEXT NOT NULL,
 payload TEXT NOT NULL CHECK(length(CAST(payload AS BLOB))<=262144)
);
CREATE INDEX advice_record_lookup ON advice_evidence_records(product_id,available_at,report_date);
CREATE TABLE advice_evidence_links(
 id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES advice_evidence_records(id),
 capture_id TEXT NOT NULL REFERENCES advice_evidence_captures(id),
 locator TEXT NOT NULL CHECK(length(CAST(locator AS BLOB))<=8192),
 linked_at TEXT NOT NULL, UNIQUE(record_id,capture_id,locator)
);
CREATE INDEX advice_link_record ON advice_evidence_links(record_id,linked_at);
"""


FORECAST_TABLES = ("forecast_records", "forecast_outcomes")
MAX_FORECAST_BYTES = 16 * 1024 * 1024
FORECAST_SCHEMA = """
CREATE TABLE forecast_records(
 id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
 request_hash TEXT NOT NULL, generated_at TEXT NOT NULL, model_id TEXT NOT NULL,
 model_hash TEXT NOT NULL, input_hash TEXT NOT NULL, basis_hash TEXT NOT NULL,
 payload TEXT NOT NULL CHECK(length(CAST(payload AS BLOB))<=16777216),
 UNIQUE(product_id,request_hash)
);
CREATE INDEX forecast_record_lookup ON forecast_records(product_id,generated_at);
CREATE TABLE forecast_outcomes(
 id TEXT PRIMARY KEY, forecast_id TEXT NOT NULL REFERENCES forecast_records(id),
 product_id TEXT NOT NULL REFERENCES products(id), horizon INTEGER NOT NULL,
 entry_lag INTEGER NOT NULL, scored_at TEXT NOT NULL,
 payload TEXT NOT NULL CHECK(length(CAST(payload AS BLOB))<=262144),
 UNIQUE(forecast_id,horizon,entry_lag)
);
CREATE INDEX forecast_outcome_lookup ON forecast_outcomes(product_id,scored_at);
"""


class Store:
    def __init__(
        self, path: str | Path, mode: str = "demo", *, read_only: bool = False,
        initialize: bool = True,
    ) -> None:
        if mode not in {"demo", "public"}:
            raise ValidationError("数据模式只能是 demo 或 public")
        self.path = Path(path)
        self.read_only = read_only
        if read_only or not initialize:
            if not self.path.is_file():
                raise FileNotFoundError("本地数据库不存在；浏览不会创建数据库，请显式初始化")
            connection_mode = "ro" if read_only else "rw"
            self.conn = sqlite3.connect(
                self.path.resolve().as_uri() + f"?mode={connection_mode}", uri=True,
                isolation_level=None, timeout=15,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=15)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=15000")
        if read_only:
            self.conn.execute("PRAGMA query_only=ON")
        elif initialize:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            journal_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(journal)")}
            if "recorded_at" not in journal_columns:
                self.conn.execute("ALTER TABLE journal ADD COLUMN recorded_at TEXT NOT NULL DEFAULT ''")
        stored = self.conn.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()
        if stored and stored[0] != mode:
            self.conn.close()
            raise ValidationError("数据库模式不匹配；禁止混合真实与虚构数据")
        if not read_only and initialize:
            self.conn.execute("INSERT OR IGNORE INTO metadata VALUES('mode',?)", (mode,))
        self.mode = mode
        self._depth = 0

    def _has_table(self, name: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
        ).fetchone() is not None

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        depth = self._depth
        savepoint = f"nested_{depth}"
        self.conn.execute("BEGIN IMMEDIATE" if depth == 0 else f"SAVEPOINT {savepoint}")
        self._depth += 1
        try:
            yield
        except BaseException:
            if depth == 0:
                self.conn.execute("ROLLBACK")
            else:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            self.conn.execute("COMMIT" if depth == 0 else f"RELEASE SAVEPOINT {savepoint}")
        finally:
            self._depth -= 1

    def add_product(self, product: Product) -> bool:
        if product.is_demo != (self.mode == "demo"):
            raise ValidationError("虚构产品与真实公开产品必须使用不同数据库")
        existing = self.get_product(product.id)
        if existing:
            if existing != product:
                raise ValidationError("产品标识已存在且内容不同；不可悄悄修改类型或份额类别")
            return False
        self.conn.execute(
            "INSERT INTO products VALUES(?,?)", (product.id, canonical_json(asdict(product)))
        )
        return True

    def get_product(self, product_id: str) -> Product | None:
        row = self.conn.execute("SELECT payload FROM products WHERE id=?", (product_id,)).fetchone()
        return Product(**json.loads(row[0])) if row else None

    def list_products(self) -> list[Product]:
        return [Product(**json.loads(row[0])) for row in self.conn.execute(
            "SELECT payload FROM products ORDER BY id"
        )]

    def add_profile(self, profile: Profile) -> bool:
        if not self.get_product(profile.product_id):
            raise ValidationError("档案产品不存在")
        payload = canonical_json(asdict(profile))
        existing = self.conn.execute("SELECT payload FROM profiles WHERE id=?", (profile.id,)).fetchone()
        if existing:
            if existing[0] != payload:
                raise ValidationError("档案版本不可覆盖；请创建新版本标识")
            return False
        self.conn.execute("INSERT INTO profiles VALUES(?,?,?,?,?,?)", (
            profile.id, profile.product_id, profile.effective_from, profile.effective_to,
            timestamp(profile.observed_at).isoformat(), payload,
        ))
        return True

    def profiles(self, product_id: str, as_of: str) -> list[Profile]:
        point = timestamp(as_of)
        day = point.date().isoformat()
        rows = self.conn.execute(
            "SELECT payload FROM profiles WHERE product_id=? AND effective_from<=? "
            "AND (effective_to IS NULL OR effective_to>=?) ORDER BY effective_from DESC,observed_at DESC",
            (product_id, day, day),
        )
        return [profile for row in rows if timestamp((profile := Profile(**json.loads(row[0]))).observed_at) <= point]

    def add_fact(self, fact: Fact) -> bool:
        if fact.kind != "calendar" and not self.get_product(fact.product_id):
            raise ValidationError(f"事实对应产品不存在: {fact.product_id}")
        if self.mode == "public" and (fact.source.startswith("demo") or fact.quality == "demo"):
            raise ValidationError("禁止将演示事实导入真实公开数据库")
        cursor = self.conn.execute("INSERT OR IGNORE INTO facts VALUES(?,?,?,?,?,?,?)", (
            fact.id, fact.product_id, fact.kind, fact.event_date, fact.observed_at,
            fact.available_at, canonical_json(asdict(fact)),
        ))
        return cursor.rowcount == 1

    def facts(
        self, product_id: str | None = None, kind: str | None = None,
        as_of: str | None = None, start: str | None = None, end: str | None = None,
    ) -> list[Fact]:
        conditions, args = [], []
        for column, value in (("product_id", product_id), ("kind", kind)):
            if value is not None:
                conditions.append(f"{column}=?")
                args.append(value)
        for operator, value in ((">=", start), ("<=", end)):
            if value:
                date.fromisoformat(value)
                conditions.append(f"event_date{operator}?")
                args.append(value)
        sql = "SELECT payload FROM facts" + (" WHERE " + " AND ".join(conditions) if conditions else "")
        sql += " ORDER BY event_date,observed_at,id"
        point = timestamp(as_of) if as_of else None
        result = [Fact(**json.loads(row[0])) for row in self.conn.execute(sql, args)]
        return [fact for fact in result if not point or timestamp(fact.available_at) <= point]

    def latest_fact(
        self, product_id: str, kind: str, as_of: str, event_date: str | None = None,
    ) -> Fact | None:
        end = event_date or timestamp(as_of).date().isoformat()
        values = self.facts(product_id, kind, as_of, start=event_date, end=end)
        return values[-1] if values else None

    def record_snapshot(
        self, source: str, product_id: str, payload: Any, observed_at: str,
        source_url: str = "", status: str = "ok", warnings: list[str] | None = None,
    ) -> str:
        observed_at = timestamp(observed_at).isoformat()
        key = digest({"source": source, "product_id": product_id, "at": observed_at,
                      "payload": payload, "status": status})
        self.conn.execute("INSERT OR IGNORE INTO snapshots VALUES(?,?,?,?,?,?,?,?,?)", (
            key, source, product_id, observed_at, source_url, status,
            canonical_json(warnings or []), digest(payload), canonical_json(payload),
        ))
        return key

    def snapshots(self, limit: int = 100) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(
            "SELECT id,source,product_id,observed_at,source_url,status,warnings,content_hash "
            "FROM snapshots ORDER BY observed_at DESC LIMIT ?", (limit,)
        )]

    def product_snapshots(
        self, product_id: str, as_of: str, *, limit: int = 32, offset: int = 0,
        usable_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Product-scoped metadata, never a global latest-100 freshness proxy."""
        if not 1 <= limit <= 128 or offset < 0:
            raise ValidationError("快照读取范围无效")
        point = timestamp(as_of).isoformat()
        status_sql = " AND status IN ('ok','partial','cached')" if usable_only else ""
        return [dict(row) for row in self.conn.execute(
            "SELECT id,source,product_id,observed_at,source_url,status,warnings,content_hash, "
            "length(CAST(payload AS BLOB)) AS payload_bytes "
            "FROM snapshots WHERE product_id=? AND observed_at<=?" + status_sql +
            " ORDER BY observed_at DESC,rowid DESC LIMIT ? OFFSET ?",
            (product_id, point, limit, offset),
        )]

    def snapshot_payload(self, snapshot_id: str, *, max_bytes: int = 16 * 1024 * 1024) -> Any:
        """Read bounded JSON, excluding an oversized BLOB before fetching its text.

        No downloaded JavaScript/HTML is evaluated.  This deliberately leaves the
        immutable snapshot schema unchanged.
        """
        if not 1 <= max_bytes <= 16 * 1024 * 1024:
            raise ValidationError("原始快照读取上限无效")
        row = self.conn.execute(
            "SELECT length(CAST(payload AS BLOB)),CASE WHEN length(CAST(payload AS BLOB))<=? "
            "THEN payload ELSE NULL END FROM snapshots WHERE id=?", (max_bytes, snapshot_id),
        ).fetchone()
        if not row:
            raise ValidationError("原始快照不存在")
        if row[0] > max_bytes:
            raise ValidationError("原始快照超过有界读取上限")
        try:
            return json.loads(row[1], parse_constant=lambda value: _reject_json_constant(value))
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValidationError("原始快照不是有效的有限 JSON") from exc

    def advice_check(
        self, product_id: str, policy_hash: str, check_month: str,
    ) -> dict[str, Any] | None:
        date.fromisoformat(check_month + "-01")
        return self._read_advice_check(
            "WHERE product_id=? AND policy_hash=? AND check_month=?",
            (product_id, policy_hash, check_month),
        )

    def latest_advice_check(
        self, product_id: str, policy_hash: str, as_of: str,
    ) -> dict[str, Any] | None:
        return self._read_advice_check(
            "WHERE product_id=? AND policy_hash=? AND generated_at<=? "
            "ORDER BY check_month DESC,generated_at DESC LIMIT 1",
            (product_id, policy_hash, timestamp(as_of).isoformat()),
        )

    def _read_advice_check(self, clause: str, args: tuple[Any, ...]) -> dict[str, Any] | None:
        # A receipt has 61 observations and compact marker evidence, never raw
        # multi-year downloads.  Bound both SQL retrieval and JSON decoding.
        if not self._has_table("advice_checks"):
            return None
        max_bytes = 256 * 1024
        row = self.conn.execute(
            "SELECT length(CAST(payload AS BLOB)),CASE WHEN length(CAST(payload AS BLOB))<=? "
            "THEN payload ELSE NULL END FROM advice_checks " + clause,
            (max_bytes, *args),
        ).fetchone()
        if not row:
            return None
        if row[0] > max_bytes:
            raise ValidationError("月度建议回执超过读取上限")
        try:
            payload = json.loads(row[1], parse_constant=lambda value: _reject_json_constant(value))
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValidationError("月度建议回执损坏；未重建或覆盖") from exc
        if not isinstance(payload, dict):
            raise ValidationError("月度建议回执结构无效；未重建或覆盖")
        return payload

    def has_evidence_schema(self) -> bool:
        """Pure existence check for raw public-document tables, not old checkers."""
        return all(self._has_table(table) for table in EVIDENCE_TABLES)

    def has_fund_review_schema(self) -> bool:
        """Compatibility name for retained public-document import providers."""
        return self.has_evidence_schema()

    def initialize_fund_review_schema(self) -> None:
        """Compatibility entry point: initialize evidence only, never a checker."""
        self.initialize_evidence_schema()

    def initialize_evidence_schema(self) -> None:
        """Explicit additive migration after a consistent SQLite backup.

        Backup on a connection in a write transaction can hang, so it is rejected
        before even creating a backup directory. Existing business tables and
        their payloads are not rewritten.
        """
        if self.conn.in_transaction or self._depth:
            raise ValidationError("资料表初始化必须在事务外显式进行")
        if self.has_evidence_schema():
            return
        existing = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?,?,?)",
            EVIDENCE_TABLES,
        ).fetchall()
        if existing:
            raise ValidationError("资料表仅部分存在；拒绝猜测迁移或覆盖")
        backup_dir = self.path.parent / "fund-review-backups"
        backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        backup_path = backup_dir / f"before-review-v1-{time.time_ns()}.sqlite3"
        if backup_path.exists():
            raise ValidationError("迁移备份路径已存在")
        backup = sqlite3.connect(backup_path)
        try:
            self.conn.backup(backup)
            if backup.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValidationError("迁移前SQLite一致性备份校验失败")
        finally:
            backup.close()
        with self.transaction():
            # A concurrent explicit migrator may have won after our backup.
            if self.has_evidence_schema():
                return
            for statement in EVIDENCE_SCHEMA.split(";"):
                if statement.strip():
                    self.conn.execute(statement)
            for table in EVIDENCE_TABLES:
                for operation in ("UPDATE", "DELETE"):
                    self.conn.execute(
                        f"CREATE TRIGGER {table}_immutable_{operation.lower()} BEFORE {operation} "
                        f"ON {table} BEGIN SELECT RAISE(ABORT,'fund review evidence is immutable'); END"
                    )
                # SQLite REPLACE may delete a conflicting row without firing its
                # delete trigger. Guard conflicting inserts as well.
                condition = "id=NEW.id"
                if table == "advice_evidence_links":
                    condition += " OR (record_id=NEW.record_id AND capture_id=NEW.capture_id AND locator=NEW.locator)"
                self.conn.execute(
                    f"CREATE TRIGGER {table}_immutable_replace BEFORE INSERT ON {table} "
                    f"WHEN EXISTS(SELECT 1 FROM {table} WHERE {condition}) "
                    "BEGIN SELECT RAISE(ABORT,'fund review evidence is immutable'); END"
                )

    def _require_fund_review_schema(self) -> None:
        if not self.has_fund_review_schema():
            raise ValidationError("基金复核资料表尚未显式初始化")

    def _review_product(self, product_id: Any) -> Product:
        if not isinstance(product_id, str) or not product_id or len(product_id) > 80:
            raise ValidationError("基金复核产品标识无效")
        product = self.get_product(product_id)
        if product is None:
            raise ValidationError("基金复核产品不存在")
        return product

    @staticmethod
    def _decode_review_json(payload: str) -> dict:
        try:
            value = json.loads(payload, parse_constant=_reject_json_constant)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValidationError("基金复核资料不是有效有限JSON；未重建或覆盖") from exc
        if not isinstance(value, dict):
            raise ValidationError("基金复核资料必须是对象")
        return value

    def _review_payload(
        self, table: str, clause: str, args: tuple, *, max_bytes: int = MAX_RECORD_BYTES,
    ) -> dict | None:
        if table not in FUND_REVIEW_TABLES or table == "advice_evidence_links":
            raise ValidationError("基金复核读取表无效")
        row = self.conn.execute(
            "SELECT length(CAST(payload AS BLOB)),CASE WHEN length(CAST(payload AS BLOB))<=? "
            f"THEN payload ELSE NULL END FROM {table} " + clause, (max_bytes, *args),
        ).fetchone()
        if row is None:
            return None
        if row[0] > max_bytes:
            raise ValidationError("基金复核资料超过有界读取上限")
        return self._decode_review_json(row[1])

    def record_advice_capture(self, capture: dict) -> str:
        self._require_fund_review_schema()
        value = normalize_capture(capture)
        self._review_product(value["product_id"])
        if self.mode == "public" and any(value[key].lower().startswith("demo") for key in ("source", "upstream")):
            raise ValidationError("禁止将演示资料写入真实公开资料表")
        metadata = {key: item for key, item in value.items() if key != "raw_bytes"}
        payload = bounded_json(metadata)
        with self.transaction():
            existing = self._review_payload("advice_evidence_captures", "WHERE id=?", (value["id"],))
            if existing is not None:
                if existing != metadata:
                    raise ValidationError("采集标识冲突；不可覆盖")
                return value["id"]
            self.conn.execute(
                "INSERT INTO advice_evidence_captures "
                "(id,product_id,family,source,upstream,source_url,observed_at,available_at,status,raw_sha256,raw_bytes,payload) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (*(value[key] for key in ("id", "product_id", "family", "source", "upstream", "source_url", "observed_at", "available_at", "status", "raw_sha256", "raw_bytes")), payload),
            )
        return value["id"]

    def advice_capture(self, capture_id: str, *, max_bytes: int = MAX_CAPTURE_BYTES) -> dict | None:
        """Bounded original bytes for explicit source/CLI diagnostics, never executed."""
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_CAPTURE_BYTES:
            raise ValidationError("原始资料读取上限无效")
        if not self.has_fund_review_schema():
            return None
        metadata = self._review_payload("advice_evidence_captures", "WHERE id=?", (capture_id,))
        if metadata is None:
            return None
        row = self.conn.execute(
            "SELECT length(raw_bytes),CASE WHEN length(raw_bytes)<=? THEN raw_bytes ELSE NULL END "
            "FROM advice_evidence_captures WHERE id=?", (max_bytes, capture_id),
        ).fetchone()
        if row[0] > max_bytes:
            raise ValidationError("原始资料超过有界读取上限")
        if hashlib.sha256(row[1]).hexdigest() != metadata.get("raw_sha256"):
            raise ValidationError("原始资料字节哈希不匹配")
        return {**metadata, "raw_bytes": row[1]}

    def advice_captures(self, product_id: str, as_of: str, *, limit: int = 32) -> list[dict]:
        """Product-local bounded metadata, including failed captures, without bytes."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 128:
            raise ValidationError("采集诊断读取条数无效")
        point = timestamp(as_of).isoformat()
        if not self.has_fund_review_schema():
            return []
        rows = self.conn.execute(
            "SELECT id,length(CAST(payload AS BLOB)) AS size FROM advice_evidence_captures "
            "WHERE product_id=? AND observed_at<=? AND available_at<=? "
            "ORDER BY observed_at DESC,id DESC LIMIT ?", (product_id, point, point, limit),
        ).fetchall()
        if sum(row["size"] for row in rows) > MAX_EVIDENCE_READ_BYTES:
            raise ValidationError("采集诊断总字节超过读取上限")
        return [self._review_payload("advice_evidence_captures", "WHERE id=?", (row["id"],)) for row in rows]

    def add_advice_evidence(self, record: dict, capture_id: str, locator: str) -> dict:
        self._require_fund_review_schema()
        value = normalize_record(record)
        product = self._review_product(value["product_id"])
        if not isinstance(locator, str) or not locator.strip() or len(locator.encode("utf-8")) > 8192 or any(ord(char) < 32 for char in locator):
            raise ValidationError("资料必须有有界文档页/表/行定位")
        if value["family"] == "mandate":
            mandate = value["payload"]
            if mandate["symbol"] != product.symbol or (product.share_class and mandate["share_class"] != product.share_class):
                raise ValidationError("投资范围资料与实际产品代码/份额类别不匹配")
        if value["family"] == "research_thesis" and any(
            value["payload"]["product"][key] != getattr(product, key)
            for key in ("id", "symbol", "name", "kind", "share_class", "exchange")
        ):
            raise ValidationError("研究论题必须匹配实际登记产品的完整身份")
        with self.transaction():
            capture = self._review_payload("advice_evidence_captures", "WHERE id=?", (capture_id,))
            if capture is None or capture.get("status") not in {"ok", "reviewed"}:
                raise ValidationError("资料必须链接成功取得或已复核的实际采集")
            pairing = ("product_id", "family", "source", "upstream", "source_url")
            if any(capture.get(key) != value[key] for key in pairing):
                raise ValidationError("资料与采集的产品/家族/来源/上游/URL不匹配")
            existing = self._review_payload("advice_evidence_records", "WHERE id=?", (value["id"],))
            if existing is None:
                if timestamp(value["observed_at"]) < timestamp(capture["observed_at"]) or timestamp(value["available_at"]) < timestamp(capture["available_at"]):
                    raise ValidationError("首次资料不能补造早于实际采集/复核的可用时间")
                self.conn.execute(
                    "INSERT INTO advice_evidence_records "
                    "(id,product_id,family,source,upstream,source_url,report_date,observed_at,available_at,payload) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (*(value[key] for key in ("id", *pairing, "report_date", "observed_at", "available_at")), bounded_json(value)),
                )
                existing = value
            else:
                original = normalize_record(existing)
                comparison = value
                if value["family"] == "research_thesis":
                    original = {**original, "payload": research_thesis_content(original["payload"])}
                    comparison = {**value, "payload": research_thesis_content(value["payload"])}
                if any(original[key] != comparison[key] for key in comparison if key not in {"observed_at", "available_at"}):
                    raise ValidationError("资料内容标识冲突；不可覆盖")
            # A refetch may reuse the original record dict. Its new capture check
            # still begins at the actual later capture/review, never in the past.
            linked_at = max(timestamp(capture["available_at"]), timestamp(value["available_at"]), timestamp(existing["available_at"])).isoformat()
            link_id = digest({"record_id": value["id"], "capture_id": capture_id, "locator": locator})
            link = self.conn.execute("SELECT linked_at FROM advice_evidence_links WHERE id=?", (link_id,)).fetchone()
            if link is None:
                self.conn.execute("INSERT INTO advice_evidence_links VALUES(?,?,?,?,?)", (link_id, value["id"], capture_id, locator, linked_at))
            return existing

    def advice_evidence(self, product_id: str, as_of: str) -> list[dict]:
        """All available history, bounded before JSON retrieval, with as-of checks.

        Later linked captures never renew a historical check. Failed captures do
        not count, and no family is silently truncated when the budget is hit.
        """
        point = timestamp(as_of).isoformat()
        if not self.has_fund_review_schema():
            return []
        pairing = " AND ".join(f"c.{key}=r.{key}" for key in ("product_id", "family", "source", "upstream", "source_url"))
        eligibility = (
            " FROM advice_evidence_records r JOIN advice_evidence_links l ON l.record_id=r.id "
            "JOIN advice_evidence_captures c ON c.id=l.capture_id "
            "WHERE r.product_id=? AND r.available_at<=? AND r.observed_at<=? "
            "AND l.linked_at<=? AND c.observed_at<=? AND c.available_at<=? "
            "AND c.status IN ('ok','reviewed') AND " + pairing
        )
        args = (product_id, point, point, point, point, point)
        rows = self.conn.execute(
            "SELECT r.id,length(CAST(r.payload AS BLOB)) AS size,MAX(l.linked_at) AS checked_at,"
            "COUNT(*) AS link_count,SUM(length(CAST(l.locator AS BLOB))) AS locator_size" + eligibility +
            " GROUP BY r.id ORDER BY r.report_date,r.available_at,r.id LIMIT ?",
            (*args, MAX_EVIDENCE_RECORDS + 1),
        ).fetchall()
        if len(rows) > MAX_EVIDENCE_RECORDS or sum(row["link_count"] for row in rows) > MAX_EVIDENCE_LINKS:
            raise ValidationError("资料历史/采集链接条数超过读取上限；未截断风险证据")
        if sum(row["size"] + row["locator_size"] for row in rows) > MAX_EVIDENCE_READ_BYTES:
            raise ValidationError("资料历史总字节超过读取上限；未截断风险证据")
        result = []
        for row in rows:
            value = self._review_payload("advice_evidence_records", "WHERE id=?", (row["id"],))
            value = normalize_record(value)
            if value["id"] != row["id"]:
                raise ValidationError("资料行与经济标识不一致")
            result.append({**value, "checked_at": row["checked_at"]})
        return result

    def advice_evidence_links(self, product_id: str, record_id: str, as_of: str) -> list[dict]:
        """Bounded as-of citation links, never original bytes or a schema change.

        Locators are evidence only for the exact existing record/capture pair.
        Later captures, failed captures and another product's references cannot
        authenticate a citation in an earlier local research note.
        """
        point = timestamp(as_of).isoformat()
        if not isinstance(record_id, str) or len(record_id) != 64 or set(record_id) - set("0123456789abcdef"):
            raise ValidationError("引用资料标识必须是64位小写SHA-256")
        if not self.has_fund_review_schema():
            return []
        pairing = " AND ".join(f"c.{key}=r.{key}" for key in ("product_id", "family", "source", "upstream", "source_url"))
        eligibility = (
            " FROM advice_evidence_records r JOIN advice_evidence_links l ON l.record_id=r.id "
            "JOIN advice_evidence_captures c ON c.id=l.capture_id "
            "WHERE r.product_id=? AND r.id=? AND r.available_at<=? AND r.observed_at<=? "
            "AND l.linked_at<=? AND c.observed_at<=? AND c.available_at<=? "
            "AND c.status IN ('ok','reviewed') AND " + pairing
        )
        args = (product_id, record_id, point, point, point, point, point)
        sizes = self.conn.execute(
            "SELECT length(CAST(l.locator AS BLOB))+length(CAST(r.source_url AS BLOB)) AS size" +
            eligibility + " ORDER BY l.linked_at,l.id LIMIT ?", (*args, MAX_EVIDENCE_LINKS + 1),
        ).fetchall()
        if len(sizes) > MAX_EVIDENCE_LINKS:
            raise ValidationError("引用采集链接超过有界读取上限")
        if sum(row["size"] + 1024 for row in sizes) > MAX_EVIDENCE_READ_BYTES:
            raise ValidationError("引用采集链接总字节超过有界读取上限")
        rows = self.conn.execute(
            "SELECT l.id AS link_id,l.record_id,l.capture_id,l.locator,l.linked_at,"
            "r.family,r.source_url,c.raw_sha256,c.observed_at AS capture_observed_at,"
            "c.available_at AS capture_available_at,c.status" + eligibility +
            " ORDER BY l.linked_at,l.id LIMIT ?", (*args, MAX_EVIDENCE_LINKS + 1),
        ).fetchall()
        if len(rows) > MAX_EVIDENCE_LINKS:
            raise ValidationError("引用采集链接超过有界读取上限")
        result = [dict(row) for row in rows]
        if sum(len(bounded_json(row).encode("utf-8")) for row in result) > MAX_EVIDENCE_READ_BYTES:
            raise ValidationError("引用采集链接总字节超过有界读取上限")
        return result

    def daily_fund_review(self, product_id: str, policy_hash: str, check_date: str) -> dict | None:
        _review_day(check_date)
        if not self._has_table("daily_advice_checks"):
            return None
        return self._review_payload("daily_advice_checks", "WHERE product_id=? AND policy_hash=? AND check_date=?", (product_id, policy_hash, check_date))

    def latest_fund_review(self, product_id: str, policy_hash: str, as_of: str) -> dict | None:
        point = timestamp(as_of).isoformat()
        if not self._has_table("daily_advice_checks"):
            return None
        return self._review_payload(
            "daily_advice_checks", "WHERE product_id=? AND policy_hash=? AND generated_at<=? "
            "ORDER BY check_date DESC,generated_at DESC,id DESC LIMIT 1", (product_id, policy_hash, point),
        )

    def fund_review_activation(self, product_id: str, as_of: str) -> dict | None:
        point = timestamp(as_of).isoformat()
        if not self._has_table("advice_policy_activations"):
            return None
        # Historical archive only; no current permission is derived from it.
        return self._review_payload(
            "advice_policy_activations", "WHERE product_id=? AND recorded_at<=? "
            "ORDER BY recorded_at DESC,rowid DESC LIMIT 1", (product_id, point),
        )
    def has_forecast_schema(self) -> bool:
        return all(self._has_table(table) for table in FORECAST_TABLES)

    def initialize_forecast_schema(self) -> None:
        """Lazy additive schema for explicit forecast writes only."""
        if self.read_only:
            raise ValidationError("只读数据库不能保存预测")
        with self.transaction():
            if self.has_forecast_schema():
                return
            if any(self._has_table(table) for table in FORECAST_TABLES):
                raise ValidationError("预测表仅部分存在；未猜测修复")
            for statement in FORECAST_SCHEMA.split(";"):
                if statement.strip():
                    self.conn.execute(statement)
            for table, unique in (
                ("forecast_records", "product_id=NEW.product_id AND request_hash=NEW.request_hash"),
                ("forecast_outcomes", "forecast_id=NEW.forecast_id AND horizon=NEW.horizon AND entry_lag=NEW.entry_lag"),
            ):
                for operation in ("UPDATE", "DELETE"):
                    self.conn.execute(
                        f"CREATE TRIGGER {table}_immutable_{operation.lower()} BEFORE {operation} "
                        f"ON {table} BEGIN SELECT RAISE(ABORT,'forecast records are immutable'); END"
                    )
                self.conn.execute(
                    f"CREATE TRIGGER {table}_immutable_replace BEFORE INSERT ON {table} "
                    f"WHEN EXISTS(SELECT 1 FROM {table} WHERE id=NEW.id OR ({unique})) "
                    "BEGIN SELECT RAISE(ABORT,'forecast records are immutable'); END"
                )

    def _forecast_payload(self, table: str, clause: str, args: tuple) -> dict | None:
        if table not in FORECAST_TABLES:
            raise ValidationError("预测读取表无效")
        if not self._has_table(table):
            return None
        row = self.conn.execute(
            "SELECT length(CAST(payload AS BLOB)),CASE WHEN length(CAST(payload AS BLOB))<=? "
            f"THEN payload ELSE NULL END FROM {table} " + clause,
            (MAX_FORECAST_BYTES, *args),
        ).fetchone()
        if row is None:
            return None
        if row[0] > MAX_FORECAST_BYTES:
            raise ValidationError("预测记录超过读取上限")
        return self._decode_review_json(row[1])

    def factor_capture(self, product_id: str, *, as_of: str, usable_only: bool = False) -> dict | None:
        """Read a saved public input, without creating its table or acquiring data."""
        if not self._has_table("investment_factor_captures"):
            return None
        status = " AND status='ready'" if usable_only else ""
        row = self.conn.execute(
            "SELECT payload FROM investment_factor_captures WHERE product_id=? AND observed_at<=?"
            + status + " ORDER BY observed_at DESC,rowid DESC LIMIT 1",
            (product_id, timestamp(as_of).isoformat()),
        ).fetchone()
        return self._decode_review_json(row[0]) if row else None

    def add_factor_capture(self, product_id: str, result: Mapping, *, observed_at: str) -> dict:
        """Append an explicit factor acquisition/import; retain unchanged first-seen rows."""
        from .investment_factors import factor_input_hash, normalize_factor_bundle

        if self.read_only:
            raise ValidationError("只读数据库不能保存投资因素")
        self._review_product(product_id)
        point = timestamp(observed_at).isoformat()
        previous = self.factor_capture(product_id, as_of=point, usable_only=True)
        bundle = result.get("bundle")
        if bundle is not None:
            bundle = normalize_factor_bundle(bundle, as_of=point)
            if previous:
                for role, series in bundle["series"].items():
                    prior = previous["bundle"]["series"].get(role, {})
                    old = {row["date"]: row for row in prior.get("rows", [])}
                    identity = ("series_id", "currency", "unit", "return_basis", "source",
                                "version_id", "source_version")
                    old_url = urlsplit(prior.get("source_url", ""))
                    new_url = urlsplit(series["source_url"])
                    same_source = (old_url.scheme.lower(), old_url.netloc.lower(), old_url.path) == (
                        new_url.scheme.lower(), new_url.netloc.lower(), new_url.path)
                    if same_source and all(prior.get(key) == series.get(key) for key in identity):
                        metadata = series.get("metadata", {})
                        if metadata.get("complete_window") is True:
                            start = date.fromisoformat(metadata["coverage_start"]).isoformat()
                            end = date.fromisoformat(metadata["coverage_end"]).isoformat()
                            if start > end:
                                raise ValidationError("投资因素来源覆盖区间倒置")
                            # Null/withdrawn observations inside this complete page window
                            # must disappear; bounded older history may still be retained.
                            merged = {day: row for day, row in old.items() if not start <= day <= end}
                        else:
                            merged = dict(old)
                        for row in series["rows"]:
                            existing = old.get(row["date"])
                            unchanged = existing is not None and all(existing.get(key) == row.get(key)
                                for key in ("value", "version_id", "fact_id"))
                            merged[row["date"]] = existing if unchanged else row
                        series["rows"] = [merged[day] for day in sorted(merged)]
            input_hash = factor_input_hash(bundle)
        else:
            input_hash = None
        value = {"product_id": product_id, "observed_at": point,
                 "status": result.get("status", "ready" if bundle else "error"),
                 "bundle": bundle, "input_hash": input_hash,
                 "warnings": list(result.get("warnings", [])),
                 "captures": list(result.get("captures", []))}
        _reject_personal_fields(value)
        key = digest(value)
        value["id"] = key
        _, payload = _forecast_json(value)
        with self.transaction():
            self.conn.execute("CREATE TABLE IF NOT EXISTS investment_factor_captures("
                "id TEXT PRIMARY KEY,product_id TEXT NOT NULL REFERENCES products(id),"
                "observed_at TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL)")
            self.conn.execute("INSERT OR IGNORE INTO investment_factor_captures VALUES(?,?,?,?,?)",
                              (key, product_id, point, value["status"], payload))
        return value

    def forecast_record(
        self, product_id: str, *, as_of: str, request_hash: str | None = None,
        model_id: str | None = None,
    ) -> dict | None:
        point = timestamp(as_of).isoformat()
        clause = "WHERE product_id=? AND generated_at<=?"
        args: tuple = (product_id, point)
        if request_hash is not None:
            clause += " AND request_hash=?"
            args += (request_hash,)
        if model_id is not None:
            if model_id in {"nav_ridge_empirical_v1", "nav_ridge_empirical_v2"}:
                clause += " AND model_id IN ('nav_ridge_empirical_v1','nav_ridge_empirical_v2')"
            else:
                clause += " AND model_id=?"
                args += (model_id,)
        return self._forecast_payload(
            "forecast_records", clause + " ORDER BY generated_at DESC,rowid DESC LIMIT 1", args,
        )

    def forecast_records(self, product_id: str, as_of: str) -> list[dict]:
        if not self._has_table("forecast_records"):
            return []
        rows = self.conn.execute(
            "SELECT id FROM forecast_records WHERE product_id=? AND generated_at<=? "
            "ORDER BY generated_at,id", (product_id, timestamp(as_of).isoformat()),
        ).fetchall()
        return [self._forecast_payload("forecast_records", "WHERE id=?", (row[0],)) for row in rows]

    def forecast_history_heads(
        self, product_id: str, as_of: str, *, model_id: str | None = None,
        horizon: int | None = None, limit: int = 30, offset: int = 0,
        role: str = "anchor", entry_lag: int | None = None,
    ) -> list[dict]:
        """Page original heads, retaining the legacy anchor-only default.

        A lag-0 execution alias belongs to both roles but is one physical head.
        Role membership comes from the issue's saved assumptions, never today's
        assumptions or the smallest available execution lag.
        """
        if (isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 128
                or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0):
            raise ValidationError("预测历史分页范围无效")
        if role not in {"anchor", "primary", "stress"}:
            raise ValidationError("预测历史情景类别无效")
        return self._forecast_head_rows(
            product_id, as_of, model_id=model_id, horizon=horizon,
            role=role, entry_lag=entry_lag, limit=limit, offset=offset,
        )

    def forecast_evidence_heads(
        self, product_id: str, as_of: str, *, horizon: int | None = None,
        model_id: str | None = None,
    ) -> list[dict]:
        """All issued heads as of an exact instant, not just a history page.

        SQL strips both predictive and baseline samples before JSON retrieval;
        the potentially large input series and historical backtests never leave
        SQLite. An outcome is included only if it was already saved by as_of.
        No missing head, baseline, assumption or outcome is reconstructed.
        """
        return self._forecast_head_rows(product_id, as_of, model_id=model_id, horizon=horizon)

    def _forecast_head_rows(
        self, product_id: str, as_of: str, *, model_id: str | None = None,
        horizon: int | None = None, role: str | None = None,
        entry_lag: int | None = None, limit: int | None = None, offset: int = 0,
    ) -> list[dict]:
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0):
            raise ValidationError("预测历史期限必须是正整数")
        if entry_lag is not None and (isinstance(entry_lag, bool) or not isinstance(entry_lag, int) or entry_lag < 0):
            raise ValidationError("预测历史延迟必须是非负整数")
        if model_id is not None:
            _review_identifier(model_id)
        point = timestamp(as_of).isoformat()
        if not self._has_table("forecast_records"):
            return []
        conditions = ["r.product_id=?", "r.generated_at<=?", "h.type='object'",
                      "json_extract(h.value,'$.status')='available'"]
        args: list[Any] = [product_id, point]
        if model_id is not None:
            if model_id in {"nav_ridge_empirical_v1", "nav_ridge_empirical_v2"}:
                conditions.append("r.model_id IN ('nav_ridge_empirical_v1','nav_ridge_empirical_v2')")
            else:
                conditions.append("r.model_id=?")
                args.append(model_id)
        if horizon is not None:
            conditions.append("json_extract(h.value,'$.horizon')=?")
            args.append(horizon)
        if entry_lag is not None:
            conditions.append("COALESCE(json_extract(h.value,'$.entry_lag'),0)=?")
            args.append(entry_lag)
        outcome_sql, outcome_column = "", "NULL"
        if self._has_table("forecast_outcomes"):
            outcome_sql = (
                " LEFT JOIN forecast_outcomes o ON o.forecast_id=h.record_id "
                "AND o.product_id=h.product_id AND o.horizon=h.horizon "
                "AND o.entry_lag=h.entry_lag AND o.scored_at<=?"
            )
            outcome_column = "o.payload"
            args.append(point)
        # Keep aliases in SQL only long enough to determine their role membership.
        # Selecting the original anchor copy is deterministic and unrelated to
        # outcomes, probabilities or whether a baseline happened to be saved.
        query = """
            WITH source_heads AS (
                SELECT r.id AS record_id,r.product_id,r.generated_at,r.model_id,r.model_hash,
                    r.input_hash,
                    json_quote(COALESCE(json_extract(h.value,'$.target_definition'),
                        json_extract(r.payload,'$.prediction.target_definition'),
                        json_extract(r.payload,'$.prediction.model_definition.target'))) AS target_definition,
                    json_extract(r.payload,'$.prediction.anchor') AS anchor,
                    CASE WHEN json_type(r.payload,'$.prediction.entry_assumptions.baseline_entry_lag')='integer'
                        AND json_extract(r.payload,'$.prediction.entry_assumptions.baseline_entry_lag')>=0
                        THEN json_extract(r.payload,'$.prediction.entry_assumptions.baseline_entry_lag')
                        ELSE NULL END AS baseline_entry_lag,
                    json_extract(h.value,'$.horizon') AS horizon,
                    COALESCE(json_extract(h.value,'$.entry_lag'),0) AS entry_lag,
                    json_remove(h.value,'$.samples','$.historical_baseline.samples') AS head,
                    collections.value AS collection,CAST(h.key AS INTEGER) AS head_order
                FROM forecast_records r
                JOIN json_each('["forecasts","execution_forecasts"]') collections
                JOIN json_each(r.payload,'$.prediction.' || collections.value) h
                WHERE """ + " AND ".join(conditions) + """
            ), heads AS (
                SELECT *,
                    MAX(collection='forecasts') OVER identity AS is_anchor,
                    MAX(collection='execution_forecasts' AND entry_lag=baseline_entry_lag)
                        OVER identity AS is_primary,
                    MAX(collection='execution_forecasts' AND baseline_entry_lag IS NOT NULL
                        AND entry_lag!=baseline_entry_lag) OVER identity AS is_stress,
                    ROW_NUMBER() OVER (PARTITION BY record_id,horizon,entry_lag
                        ORDER BY collection='forecasts' DESC,head_order) AS head_rank
                FROM source_heads
                WINDOW identity AS (PARTITION BY record_id,horizon,entry_lag)
            )
            SELECT h.record_id,h.product_id,h.generated_at,h.model_id,h.model_hash,h.input_hash,
                h.target_definition,h.anchor,h.baseline_entry_lag,h.horizon,h.entry_lag,
                h.is_anchor,h.is_primary,h.is_stress,h.head,""" + outcome_column + " AS outcome " + (
                    "FROM heads h" + outcome_sql + " WHERE h.head_rank=1"
                )
        if role is not None:
            query += f" AND h.is_{role}=1"
        query += " ORDER BY h.generated_at DESC,h.record_id DESC,h.horizon,h.entry_lag"
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            args.extend((limit, offset))
        result = []
        for row in self.conn.execute(query, args):
            value = dict(row)
            for key in ("anchor", "head", "outcome"):
                value[key] = self._decode_review_json(value[key]) if value[key] is not None else None
            value["target_definition"] = json.loads(value["target_definition"])
            value["roles"] = [name for name in ("anchor", "primary", "stress")
                              if value.pop(f"is_{name}")]
            value["role"] = next((name for name in ("primary", "anchor", "stress")
                                  if name in value["roles"]), "unclassified")
            value["as_of"] = point
            result.append(value)
        return result

    def forecast_record_by_id(
        self, product_id: str, record_id: str, *, as_of: str,
    ) -> dict | None:
        """Fetch one original prediction, excluding the potentially 16-MB series."""
        point = timestamp(as_of).isoformat()
        if not self._has_table("forecast_records"):
            return None
        row = self.conn.execute(
            "SELECT id AS record_id,product_id,generated_at,model_id,model_hash,"
            "json_extract(payload,'$.prediction') AS prediction FROM forecast_records "
            "WHERE id=? AND product_id=? AND generated_at<=?", (record_id, product_id, point),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["prediction"] = self._decode_review_json(result["prediction"])
        return result

    def forecast_record_outcomes(self, product_id: str, record_id: str, as_of: str) -> list[dict]:
        """Only outcomes actually saved for this issue, not recomputed maturity."""
        point = timestamp(as_of).isoformat()
        if not self._has_table("forecast_outcomes"):
            return []
        return [self._decode_review_json(row[0]) for row in self.conn.execute(
            "SELECT payload FROM forecast_outcomes WHERE product_id=? AND forecast_id=? "
            "AND scored_at<=? ORDER BY horizon,entry_lag,id", (product_id, record_id, point),
        )]

    def add_forecast_record(self, record: dict) -> dict:
        required = {"id", "product_id", "request_hash", "generated_at", "model_id", "model_hash",
                    "input_hash", "basis_hash", "prediction", "series", "entry_assumptions"}
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValidationError("预测记录字段无效；不能保存个人回答")
        value, payload = _forecast_json(record)
        _reject_personal_fields(value)
        self._review_product(value["product_id"])
        for key in ("id", "request_hash", "model_id", "model_hash", "input_hash", "basis_hash"):
            _review_identifier(value[key])
        value["generated_at"] = timestamp(value["generated_at"]).isoformat()
        prediction = value["prediction"]
        if (not isinstance(prediction, dict) or not isinstance(value["series"], list)
                or not isinstance(value["entry_assumptions"], dict)
                or any(prediction.get(key) != value[key] for key in
                       ("model_id", "model_hash", "input_hash", "generated_at"))):
            raise ValidationError("预测记录与模型输出标识不一致")
        _, payload = _forecast_json(value)
        self.initialize_forecast_schema()
        with self.transaction():
            existing = self._forecast_payload(
                "forecast_records", "WHERE product_id=? AND request_hash=?",
                (value["product_id"], value["request_hash"]),
            )
            if existing is not None:
                return existing
            if self.conn.execute("SELECT 1 FROM forecast_records WHERE id=?", (value["id"],)).fetchone():
                raise ValidationError("预测标识冲突；未覆盖旧记录")
            self.conn.execute(
                "INSERT INTO forecast_records VALUES(?,?,?,?,?,?,?,?,?)",
                (*(value[key] for key in ("id", "product_id", "request_hash", "generated_at",
                                         "model_id", "model_hash", "input_hash", "basis_hash")), payload),
            )
        return value

    def forecast_outcomes(self, product_id: str, as_of: str) -> list[dict]:
        if not self._has_table("forecast_outcomes"):
            return []
        rows = self.conn.execute(
            "SELECT id FROM forecast_outcomes WHERE product_id=? AND scored_at<=? "
            "ORDER BY scored_at,id", (product_id, timestamp(as_of).isoformat()),
        ).fetchall()
        return [self._forecast_payload("forecast_outcomes", "WHERE id=?", (row[0],)) for row in rows]

    def add_forecast_outcome(self, outcome: dict) -> dict:
        required = {"id", "forecast_id", "product_id", "horizon", "entry_lag", "scored_at",
                    "issued_at", "kind", "eligible_forward", "entry", "target", "actual_change",
                    "up_probability", "brier_score", "interval_covered", "input_hash"}
        if not isinstance(outcome, Mapping) or set(outcome) != required:
            raise ValidationError("预测结果记录字段无效")
        value, payload = _forecast_json(outcome, max_bytes=MAX_RECORD_BYTES)
        _reject_personal_fields(value)
        for key in ("id", "forecast_id", "product_id", "input_hash"):
            _review_identifier(value[key])
        if (any(isinstance(value[key], bool) or not isinstance(value[key], int) for key in
                ("horizon", "entry_lag")) or value["horizon"] <= 0 or value["entry_lag"] < 0
                or value["kind"] not in {"forward", "nowcast"}
                or not isinstance(value["eligible_forward"], bool)
                or value["eligible_forward"] != (value["kind"] == "forward")):
            raise ValidationError("预测结果期限或前向分类无效")
        issued, scored = timestamp(value["issued_at"]), timestamp(value["scored_at"])
        if scored < issued:
            raise ValidationError("预测结果不能早于实际发布")
        value["issued_at"], value["scored_at"] = issued.isoformat(), scored.isoformat()
        if not self.has_forecast_schema():
            raise ValidationError("预测记录不存在；不能单独补造结果")
        record = self.conn.execute(
            "SELECT product_id,generated_at,input_hash FROM forecast_records WHERE id=?",
            (value["forecast_id"],),
        ).fetchone()
        if (record is None or record["product_id"] != value["product_id"]
                or record["generated_at"] != value["issued_at"]
                or record["input_hash"] != value["input_hash"]):
            raise ValidationError("预测结果与实际发布记录不匹配")
        heads = [self._decode_review_json(row[0]) for row in self.conn.execute(
            "SELECT json_remove(h.value,'$.samples','$.historical_baseline.samples') "
            "FROM forecast_records r "
            "JOIN json_each('[\"forecasts\",\"execution_forecasts\"]') collections "
            "JOIN json_each(r.payload,'$.prediction.' || collections.value) h "
            "WHERE r.id=? AND h.type='object' AND json_extract(h.value,'$.horizon')=? "
            "AND COALESCE(json_extract(h.value,'$.entry_lag'),0)=? "
            "AND json_extract(h.value,'$.status')='available'",
            (value["forecast_id"], value["horizon"], value["entry_lag"]),
        )]
        if not heads:
            raise ValidationError("原发行没有相同期限和延迟的可用预测头；不能补造结果")
        try:
            entry_day, target_day = (date.fromisoformat(value[key]["date"]) for key in ("entry", "target"))
            entry_available, target_available = (
                timestamp(value[key]["provenance"]["available_at"]) for key in ("entry", "target")
            )
        except (TypeError, KeyError, ValueError) as exc:
            raise ValidationError("预测结果缺少实际净值日期和可用时间") from exc
        eligible = entry_day > issued.date() and target_day >= issued.date() and target_available > issued
        if (target_day <= entry_day or target_day > scored.date()
                or max(entry_available, target_available) > scored
                or eligible != value["eligible_forward"]):
            raise ValidationError("预测结果尚未成熟或前向分类与实际发布日期不符")
        try:
            entry_nav, target_nav = (Decimal(str(value[key]["nav"])) for key in ("entry", "target"))
            if any(not number.is_finite() or number <= 0 for number in (entry_nav, target_nav)):
                raise ValueError("invalid NAV")
            actual = float(target_nav / entry_nav - 1)
        except (TypeError, KeyError, ValueError, DecimalException, OverflowError) as exc:
            raise ValidationError("预测结果需要有限且为正的实际净值") from exc
        if not math.isfinite(actual):
            raise ValidationError("预测结果实际变化必须为有限数")
        actual_value = _forecast_number(value["actual_change"])
        probability = _forecast_number(value["up_probability"])
        brier = _forecast_number(value["brier_score"])
        if (not math.isclose(actual_value, actual, rel_tol=1e-12, abs_tol=1e-12)
                or (actual_value > 0) != (actual > 0) or (actual_value < 0) != (actual < 0)
                or not 0 <= probability <= 1
                or not math.isclose(brier, (probability - int(actual > 0)) ** 2,
                                    rel_tol=1e-12, abs_tol=1e-12)
                or not isinstance(value["interval_covered"], bool)):
            raise ValidationError("预测结果实际变化、概率或误差与实际净值不一致")
        # Aliases must agree as well: one outcome key cannot score two different
        # issued distributions. Never pick whichever copy makes the result look best.
        for head in heads:
            issued_probability = _forecast_number(head.get("up_probability"))
            lower, upper = (_forecast_number(head.get(key)) for key in ("lower", "upper"))
            if (probability != issued_probability
                    or not 0 <= issued_probability <= 1 or lower > upper
                    or value["interval_covered"] != (lower <= actual <= upper)):
                raise ValidationError("预测结果概率或区间覆盖与原发行预测头不一致")
        _, payload = _forecast_json(value, max_bytes=MAX_RECORD_BYTES)
        with self.transaction():
            previous = self._forecast_payload(
                "forecast_outcomes", "WHERE forecast_id=? AND horizon=? AND entry_lag=?",
                (value["forecast_id"], value["horizon"], value["entry_lag"]),
            )
            if previous is not None:
                return previous
            self.conn.execute(
                "INSERT INTO forecast_outcomes VALUES(?,?,?,?,?,?,?)",
                (*(value[key] for key in ("id", "forecast_id", "product_id", "horizon", "entry_lag", "scored_at")), payload),
            )
        return value


def _forecast_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("预测概率与结果必须是有限数")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValidationError("预测概率与结果必须是有限数") from exc
    if not math.isfinite(number):
        raise ValidationError("预测概率与结果必须是有限数")
    return number


def _forecast_json(value: Any, *, max_bytes: int = MAX_FORECAST_BYTES) -> tuple[dict, str]:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidationError("预测记录必须是有限的标准JSON") from exc
    if len(payload.encode("utf-8")) > max_bytes:
        raise ValidationError("预测记录超过保存上限")
    return json.loads(payload), payload


def _review_identifier(value: Any) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(char) < 32 for char in value):
        raise ValidationError("基金复核标识必须是有界非空文本")


def _review_day(value: Any) -> date:
    try:
        day = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("基金复核日期无效") from exc
    if value != day.isoformat():
        raise ValidationError("基金复核日期必须是YYYY-MM-DD")
    return day


def _reject_personal_fields(value: Any) -> None:
    private = {"holding", "holding_context", "holding_state", "pending_order", "all_confirmed", "latest_confirmation_date", "horizon_months", "can_bear_loss", "can_bear_volatility", "max_transaction_fee_rate", "personal", "personal_context", "suitability", "account_id", "balance", "purchase_cost", "session_id"}
    if isinstance(value, dict):
        if private.intersection(value) or value.get("state") in ("flat", "held"):
            raise ValidationError("共享预测不能保存个人回答或会话信息")
        for item in value.values():
            _reject_personal_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_personal_fields(item)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")
