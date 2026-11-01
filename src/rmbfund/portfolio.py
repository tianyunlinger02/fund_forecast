"""Local, private fund holdings, separate from public research and paper accounts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
from uuid import uuid4

from .domain import SUPPORTED_KINDS, ValidationError, decimal, now_iso

_IDENTITY = ("kind", "symbol", "share_class", "exchange")
_SUBJECT_FIELDS = {"symbol", "name", "kind", "share_class", "exchange", "is_demo"}
_COLUMNS = (
    "id, product_id, symbol, name, kind, share_class, exchange, is_demo, "
    "shares, cost, currency, created_at, updated_at"
)


def _text(value: Any, label: str, limit: int, *, blank: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{label}必须是文本")
    result = value.strip()
    if (not result and not blank) or len(result) > limit:
        raise ValidationError(f"{label}为空或过长")
    return result


def _holding_text(value: Any, label: str) -> str | None:
    """Keep entered decimal text, including meaningful trailing precision."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{label}必须使用十进制文本")
    value = value.strip()
    if not value:
        return None
    if decimal(value) < 0:
        raise ValidationError(f"{label}不得为负")
    return value


class Portfolio:
    """Passive handle to ``root/personal-{mode}.sqlite3``.

    ``save_fund`` replaces shares/cost; omitted or blank values mean unknown.
    Its subject accepts public Product fields, including the actual public ``id``
    (and optional category, which is not retained). The saved record's own ``id``
    is private and stable. Identity fields cannot be retargeted. The one allowed
    enrichment is binding an unassociated, unknown-class row to an actual public
    id/class with the same kind, symbol and exchange, using its explicit fund_id.
    Preserve shares/cost when binding; saving is still a replacement, not a patch.
    """

    def __init__(self, root: str | Path, mode: str = "public") -> None:
        if mode not in {"public", "demo"}:
            raise ValidationError("数据模式只能是 public 或 demo")
        self.root = Path(root)
        self.mode = mode
        self.path = self.root / f"personal-{mode}.sqlite3"

    def _subject(self, subject: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
        if not isinstance(subject, Mapping) or subject.keys() - (_SUBJECT_FIELDS | {"id", "category"}):
            raise ValidationError("subject只接受公开产品身份字段，不接受个人持仓或其他资料")
        symbol = _text(subject.get("symbol"), "基金代码", 30)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", symbol):
            raise ValidationError("代码仅接受字母、数字、点、短横线和下划线")
        kind = _text(subject.get("kind", "open_fund"), "产品类型", 30)
        if kind not in SUPPORTED_KINDS:
            raise ValidationError("不支持的产品类型")
        is_demo = subject.get("is_demo", self.mode == "demo")
        if not isinstance(is_demo, bool) or is_demo != (self.mode == "demo"):
            raise ValidationError("演示与真实个人持仓必须使用不同数据模式")
        name = subject.get("name")
        share_class = subject.get("share_class")
        exchange = subject.get("exchange")
        name = _text("" if name is None else name, "基金名称", 200, blank=True) or symbol
        public_id = subject.get("id")
        if public_id is not None:
            public_id = _text(public_id, "公开产品标识", 80, blank=True) or None
        return {
            "symbol": symbol, "name": name, "kind": kind,
            "share_class": _text("" if share_class is None else share_class, "份额类别", 80, blank=True),
            "exchange": _text("" if exchange is None else exchange, "交易场所", 80, blank=True),
            "is_demo": is_demo,
        }, public_id

    @contextmanager
    def _connection(self, mode: str) -> Iterator[sqlite3.Connection]:
        conn = None
        try:
            conn = sqlite3.connect(
                self.path.resolve().as_uri() + f"?mode={mode}", uri=True, timeout=15,
            )
            conn.row_factory = sqlite3.Row
            yield conn
        except (sqlite3.Error, OSError) as exc:
            raise ValidationError(
                "无法读写个人持仓文件；请检查文件格式、权限或占用，未自动重建或覆盖"
            ) from exc
        finally:
            if conn is not None:
                conn.close()

    def _check_database(self, conn: sqlite3.Connection) -> None:
        modes = conn.execute("SELECT mode FROM portfolio_meta").fetchall()
        if len(modes) != 1 or modes[0][0] != self.mode:
            raise ValidationError("个人持仓文件模式不匹配；不会混用演示与真实持仓")
        conn.execute(f"SELECT {_COLUMNS} FROM funds LIMIT 0")

    def _initialize(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE portfolio_meta (mode TEXT NOT NULL)")
        conn.execute("INSERT INTO portfolio_meta VALUES (?)", (self.mode,))
        conn.execute("""
            CREATE TABLE funds (
                id TEXT PRIMARY KEY,
                product_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                share_class TEXT NOT NULL,
                exchange TEXT NOT NULL,
                is_demo INTEGER NOT NULL CHECK (is_demo IN (0, 1)),
                shares TEXT,
                cost TEXT,
                currency TEXT NOT NULL CHECK (currency = 'CNY'),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (kind, symbol, share_class, exchange)
            )
        """)

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "product_id": row["product_id"],
            "subject": {
                key: bool(row[key]) if key == "is_demo" else row[key]
                for key in ("symbol", "name", "kind", "share_class", "exchange", "is_demo")
            },
            "shares": _holding_text(row["shares"], "已保存份额"),
            "cost": _holding_text(row["cost"], "已保存成本"),
            "currency": row["currency"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    def list_funds(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._connection("ro") as conn:
            self._check_database(conn)
            rows = conn.execute(f"SELECT {_COLUMNS} FROM funds ORDER BY created_at, id").fetchall()
            return [self._record(row) for row in rows]

    def get_fund(self, fund_id: str) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        with self._connection("ro") as conn:
            self._check_database(conn)
            row = conn.execute(f"SELECT {_COLUMNS} FROM funds WHERE id=?", (fund_id,)).fetchone()
            return self._record(row) if row is not None else None

    @staticmethod
    def _preference_id(subject: Mapping[str, Any]) -> str:
        identity = {key: subject.get(key, "") for key in _IDENTITY}
        return "research-preferences-" + hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _initialize_preferences(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research_preferences (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                symbol TEXT NOT NULL,
                share_class TEXT NOT NULL,
                exchange TEXT NOT NULL,
                conditions TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                UNIQUE (kind, symbol, share_class, exchange)
            )
        """)

    def research_preferences(self, subject: Mapping[str, Any]) -> dict[str, Any] | None:
        """Read explicitly saved local conditions without creating a database/table."""
        identity, _ = self._subject(subject)
        if not self.path.exists():
            return None
        with self._connection("ro") as conn:
            self._check_database(conn)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_preferences'").fetchone() is None:
                return None
            row = conn.execute(
                "SELECT conditions,saved_at FROM research_preferences WHERE kind=? AND symbol=? AND share_class=? AND exchange=?",
                tuple(identity[key] for key in _IDENTITY),
            ).fetchone()
            if row is None:
                return None
            try:
                conditions = json.loads(row["conditions"])
            except (TypeError, ValueError) as exc:
                raise ValidationError("已保存的研究条件无法读取；未重建或覆盖") from exc
            if not isinstance(conditions, dict):
                raise ValidationError("已保存的研究条件结构无效；未重建或覆盖")
            return {"conditions": conditions, "saved_at": row["saved_at"]}

    def save_research_preferences(self, subject: Mapping[str, Any], conditions: Mapping[str, Any]) -> dict[str, Any]:
        """Save optional local research inputs only after an explicit user action."""
        identity, _ = self._subject(subject)
        if not isinstance(conditions, Mapping):
            raise ValidationError("研究条件必须是对象")
        try:
            payload = json.dumps(dict(conditions), ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("研究条件必须是有限JSON数据") from exc
        saved_at = now_iso()
        new_file = False
        try:
            if not self.path.exists():
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    self.path.touch(mode=0o600, exist_ok=False)
                    new_file = True
                except FileExistsError:
                    pass
            with self._connection("rw") as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                if new_file:
                    self._initialize(conn)
                else:
                    self._check_database(conn)
                self._initialize_preferences(conn)
                identifier = self._preference_id(identity)
                conn.execute(
                    "INSERT INTO research_preferences (id,kind,symbol,share_class,exchange,conditions,saved_at) "
                    "VALUES (?,?,?,?,?,?,?) ON CONFLICT(kind,symbol,share_class,exchange) DO UPDATE SET "
                    "id=excluded.id,conditions=excluded.conditions,saved_at=excluded.saved_at",
                    (identifier, *(identity[key] for key in _IDENTITY), payload, saved_at),
                )
                self.path.chmod(0o600)
            return {"conditions": json.loads(payload), "saved_at": saved_at}
        except OSError as exc:
            raise ValidationError("无法保存研究条件；请检查本地目录与权限") from exc

    def save_fund(
        self, subject: Mapping[str, Any], *, shares: str | None = None,
        cost: str | None = None, fund_id: str | None = None,
    ) -> dict[str, Any]:
        subject, public_id = self._subject(subject)
        shares = _holding_text(shares, "份额")
        cost = _holding_text(cost, "持仓总成本")
        if fund_id is not None:
            fund_id = _text(fund_id, "个人基金标识", 80)
            if not self.path.exists():
                raise ValidationError("要更正的个人基金记录不存在")

        # Only a fully validated save may create a directory/file. Reads/removal
        # never initialize, and an existing wrong file is never adopted or reset.
        new_file = False
        try:
            if not self.path.exists():
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    self.path.touch(mode=0o600, exist_ok=False)
                    new_file = True
                except FileExistsError:
                    pass
            with self._connection("rw") as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                if new_file:
                    self._initialize(conn)
                else:
                    self._check_database(conn)
                identity = tuple(subject[key] for key in _IDENTITY)
                if fund_id is not None:
                    current = conn.execute("SELECT * FROM funds WHERE id=?", (fund_id,)).fetchone()
                    if current is None:
                        raise ValidationError("要更正的个人基金记录不存在")
                else:
                    matches = conn.execute(
                        "SELECT * FROM funds WHERE product_id=? OR "
                        "(kind=? AND symbol=? AND share_class=? AND exchange=?)",
                        (public_id, *identity),
                    ).fetchall()
                    if len(matches) > 1:
                        raise ValidationError("公开标识与个人基金身份冲突；未合并记录")
                    current = matches[0] if matches else None
                if current is not None:
                    resolving_class = (
                        fund_id is not None and current["product_id"] is None and public_id
                        and not current["share_class"]
                        and all(current[key] == subject[key] for key in ("kind", "symbol", "exchange"))
                    )
                    if tuple(current[key] for key in _IDENTITY) != identity and not resolving_class:
                        raise ValidationError("不能把已有个人记录改成另一基金或份额类别；请移除后另加")
                    if public_id and current["product_id"] and public_id != current["product_id"]:
                        raise ValidationError("基金已经关联另一公开产品标识；未更改关联")
                    public_id = public_id or current["product_id"]
                    fund_id = current["id"]
                if conn.execute(
                    "SELECT 1 FROM funds WHERE kind=? AND symbol=? AND share_class=? "
                    "AND exchange=? AND id!=?", (*identity, fund_id or ""),
                ).fetchone():
                    raise ValidationError("该基金身份已有另一条个人记录；未合并持仓")
                if public_id and conn.execute(
                    "SELECT 1 FROM funds WHERE product_id=? AND id!=?",
                    (public_id, fund_id or ""),
                ).fetchone():
                    raise ValidationError("公开产品已关联另一条个人记录；未合并记录")

                saved_at = now_iso()
                record = {
                    "id": fund_id or f"fund-{uuid4().hex}", "product_id": public_id,
                    "subject": subject, "shares": shares, "cost": cost, "currency": "CNY",
                    "created_at": current["created_at"] if current is not None else saved_at,
                    "updated_at": saved_at,
                }
                conn.execute(
                    f"INSERT INTO funds ({_COLUMNS}) VALUES ({','.join('?' for _ in range(13))}) "
                    "ON CONFLICT(id) DO UPDATE SET product_id=excluded.product_id, "
                    "symbol=excluded.symbol, name=excluded.name, kind=excluded.kind, "
                    "share_class=excluded.share_class, exchange=excluded.exchange, "
                    "is_demo=excluded.is_demo, shares=excluded.shares, cost=excluded.cost, "
                    "updated_at=excluded.updated_at",
                    (record["id"], public_id, subject["symbol"], subject["name"], subject["kind"],
                     subject["share_class"], subject["exchange"], int(subject["is_demo"]),
                     shares, cost, "CNY", record["created_at"], saved_at),
                )
                self.path.chmod(0o600)
            return record
        except OSError as exc:
            raise ValidationError("无法保存个人持仓文件；请检查本地目录与权限") from exc

    def remove_fund(self, fund_id: str) -> None:
        if not self.path.exists():
            return
        with self._connection("rw") as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._check_database(conn)
            conn.execute("DELETE FROM funds WHERE id=?", (fund_id,))

    @staticmethod
    def _has_assessments(conn: sqlite3.Connection) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='assessments'"
        ).fetchone() is not None

    @staticmethod
    def _initialize_assessments(conn: sqlite3.Connection) -> None:
        """An explicit assessment save is the only caller that creates this table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE,
                payload_digest TEXT NOT NULL,
                product_id TEXT,
                kind TEXT NOT NULL,
                symbol TEXT NOT NULL,
                share_class TEXT NOT NULL,
                exchange TEXT NOT NULL,
                saved_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS assessments_product_time "
                     "ON assessments(product_id, saved_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS assessments_identity_time "
                     "ON assessments(kind, symbol, share_class, exchange, saved_at)")
        for action in ("UPDATE", "DELETE"):
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS assessments_no_{action.lower()}
                BEFORE {action} ON assessments
                BEGIN SELECT RAISE(ABORT, 'assessment snapshots are append-only'); END
            """)

    @staticmethod
    def _assessment_record(row: sqlite3.Row) -> dict[str, Any]:
        try:
            assessment = json.loads(row["payload"])
        except (TypeError, ValueError) as exc:
            raise ValidationError("已保存的个人判断内容无法读取；未重算或覆盖历史") from exc
        if hashlib.sha256(row["payload"].encode()).hexdigest() != row["payload_digest"]:
            raise ValidationError("已保存判断与其内容摘要不符；未重算或覆盖历史")
        return {
            "id": row["id"], "saved_at": row["saved_at"],
            "submission_id": row["submission_id"], "payload_digest": row["payload_digest"],
            "assessment": assessment,
        }

    def save_assessment(
        self, assessment: Mapping[str, Any], *, submission_id: str | None = None,
    ) -> dict[str, Any]:
        """Append the already displayed object, never recompute or add a holding.

        A repeated submission with identical content returns its original record
        and save time. Reusing its id for changed content is an error, not an
        overwrite. Without an explicit submission id, the payload digest is the
        stable idempotency key. Public forecast issuance and this local save are
        independent events with separate timestamps.
        """
        if not isinstance(assessment, Mapping):
            raise ValidationError("个人判断应为本次已展示的对象")
        subject, public_id = self._subject(assessment.get("subject"))
        if (not isinstance(assessment.get("schema_version"), int)
                or isinstance(assessment["schema_version"], bool)
                or assessment["schema_version"] < 1):
            raise ValidationError("个人判断缺少有效的schema版本")
        _text(assessment.get("policy_version"), "判断逻辑版本", 100)
        if not isinstance(assessment.get("decision"), Mapping):
            raise ValidationError("个人判断缺少当时展示的结论")
        forecast = assessment.get("forecast")
        if not isinstance(forecast, Mapping):
            raise ValidationError("个人判断缺少原公共预测身份")
        if (public_id and forecast.get("product_id") is not None
                and forecast["product_id"] != public_id):
            raise ValidationError("基金身份与原预测不匹配；未保存判断")
        try:
            payload = json.dumps(dict(assessment), ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("个人判断必须是有限JSON数据，金额请保留十进制文本") from exc
        payload_digest = hashlib.sha256(payload.encode()).hexdigest()
        submission_id = (_text(submission_id, "本次保存标识", 200)
                         if submission_id is not None else f"payload-{payload_digest}")
        new_file = False
        try:
            if not self.path.exists():
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    self.path.touch(mode=0o600, exist_ok=False)
                    new_file = True
                except FileExistsError:
                    pass
            with self._connection("rw") as conn, conn:
                conn.execute("BEGIN IMMEDIATE")
                if new_file:
                    self._initialize(conn)
                else:
                    self._check_database(conn)
                self._initialize_assessments(conn)
                existing = conn.execute(
                    "SELECT * FROM assessments WHERE submission_id=?", (submission_id,),
                ).fetchone()
                if existing is not None:
                    if existing["payload_digest"] != payload_digest or existing["payload"] != payload:
                        raise ValidationError("同一次保存标识已用于不同内容；原判断未覆盖，请重新提交")
                    return self._assessment_record(existing)
                assessment_id, saved_at = f"assessment-{uuid4().hex}", now_iso()
                conn.execute(
                    "INSERT INTO assessments (id, submission_id, payload_digest, product_id, "
                    "kind, symbol, share_class, exchange, saved_at, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (assessment_id, submission_id, payload_digest, public_id,
                     *(subject[key] for key in _IDENTITY), saved_at, payload),
                )
                row = conn.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
                self.path.chmod(0o600)
                return self._assessment_record(row)
        except OSError as exc:
            raise ValidationError("无法保存个人判断文件；请检查本地目录与权限") from exc

    def assessment_history(
        self, subject: Mapping[str, Any] | str, *, limit: int = 20, offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Read immutable snapshots by public product or explicit class identity.

        A now-registered public subject can also find its earlier unbound exact
        identity. It never inherits a different public id or another class. No
        private fund id is used, so removing/re-adding a holding changes nothing.
        """
        if (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
                or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0):
            raise ValidationError("历史条数必须为正整数，起点必须为非负整数")
        if isinstance(subject, str):
            where = "WHERE product_id=?"
            params = [_text(subject, "公开产品标识", 80)]
        else:
            identity, public_id = self._subject(subject)
            exact = "kind=? AND symbol=? AND share_class=? AND exchange=?"
            params = [identity[key] for key in _IDENTITY]
            if public_id:
                where = f"WHERE (product_id=? OR (product_id IS NULL AND {exact}))"
                params.insert(0, public_id)
            else:
                where = f"WHERE {exact}"
        if not self.path.exists():
            return []
        with self._connection("ro") as conn:
            self._check_database(conn)
            if not self._has_assessments(conn):
                return []
            rows = conn.execute(
                f"SELECT * FROM assessments {where} ORDER BY saved_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return [self._assessment_record(row) for row in rows]

    def assessment_detail(self, assessment_id: str) -> dict[str, Any] | None:
        """Return the original displayed object; ordinary browsing never creates schema."""
        assessment_id = _text(assessment_id, "个人判断标识", 100)
        if not self.path.exists():
            return None
        with self._connection("ro") as conn:
            self._check_database(conn)
            if not self._has_assessments(conn):
                return None
            row = conn.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
            return self._assessment_record(row) if row is not None else None

    @staticmethod
    def _has_action_notes(conn: sqlite3.Connection) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='assessment_action_notes'"
        ).fetchone() is not None

    @staticmethod
    def _initialize_action_notes(conn: sqlite3.Connection) -> None:
        """Only an explicit note submission creates this append-only table."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assessment_action_notes (
                id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE,
                assessment_id TEXT NOT NULL REFERENCES assessments(id),
                product_id TEXT,
                kind TEXT NOT NULL,
                action TEXT NOT NULL,
                branch TEXT NOT NULL,
                occurred_on TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                rationale TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS assessment_action_notes_time "
                     "ON assessment_action_notes(assessment_id, recorded_at)")
        for action in ("UPDATE", "DELETE"):
            conn.execute(f"""
                CREATE TRIGGER IF NOT EXISTS assessment_action_notes_no_{action.lower()}
                BEFORE {action} ON assessment_action_notes
                BEGIN SELECT RAISE(ABORT, 'assessment action notes are append-only'); END
            """)

    @staticmethod
    def _action_note_record(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in (
            "id", "submission_id", "assessment_id", "product_id", "kind", "action", "branch",
            "occurred_on", "recorded_at", "rationale",
        )}

    def save_action_note(
        self, assessment_id: str, *, kind: str, action: str, branch: str,
        occurred_on: str, rationale: str = "", submission_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a private annotation, not a trade, holding change, or outcome.

        Occurrence dates may be historical or prospective. ``recorded_at`` always
        comes from the real save clock and cannot be supplied/backdated. Repeated
        identical submissions return the original note and its capture time.
        """
        assessment_id = _text(assessment_id, "原判断标识", 100)
        if not isinstance(kind, str) or kind not in {"plan", "simulated_choice", "user_report", "observation"}:
            raise ValidationError("笔记类别须为计划、模拟采用、用户自述或观察")
        actions = {
            "flat": {"buy", "wait", "no_action"},
            "held": {"hold", "add", "reduce50", "exit", "reduce", "redeem", "no_action"},
        }
        if (not isinstance(branch, str) or branch not in actions
                or not isinstance(action, str) or action not in actions[branch]):
            raise ValidationError("笔记动作必须属于原判断的新买或已有份额分支")
        try:
            if date.fromisoformat(occurred_on).isoformat() != occurred_on:
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise ValidationError("笔记发生日期须为实际YYYY-MM-DD日期") from exc
        rationale = _text(rationale, "简短理由或观察", 1000, blank=True)
        content = {"assessment_id": assessment_id, "kind": kind, "action": action,
                   "branch": branch, "occurred_on": occurred_on, "rationale": rationale}
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        submission_id = (_text(submission_id, "本次笔记标识", 200) if submission_id is not None
                         else "note-" + hashlib.sha256(payload.encode()).hexdigest())
        if not self.path.exists():
            raise ValidationError("原判断不存在；请先显式保存判断，未创建个人文件或笔记表")
        with self._connection("rw") as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._check_database(conn)
            original = (conn.execute("SELECT * FROM assessments WHERE id=?", (assessment_id,)).fetchone()
                        if self._has_assessments(conn) else None)
            if original is None:
                raise ValidationError("原判断不存在；未保存笔记或创建笔记表")
            self._assessment_record(original)
            self._initialize_action_notes(conn)
            previous = conn.execute(
                "SELECT * FROM assessment_action_notes WHERE submission_id=?", (submission_id,),
            ).fetchone()
            if previous is not None:
                if any(previous[key] != value for key, value in content.items()):
                    raise ValidationError("同一次笔记标识已用于不同内容；原笔记未覆盖，请重新提交")
                return self._action_note_record(previous)
            identifier, recorded_at = f"action-note-{uuid4().hex}", now_iso()
            conn.execute(
                "INSERT INTO assessment_action_notes "
                "(id,submission_id,assessment_id,product_id,kind,action,branch,occurred_on,recorded_at,rationale) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (identifier, submission_id, assessment_id, original["product_id"], kind, action,
                 branch, occurred_on, recorded_at, rationale),
            )
            row = conn.execute(
                "SELECT * FROM assessment_action_notes WHERE id=?", (identifier,),
            ).fetchone()
            self.path.chmod(0o600)
            return self._action_note_record(row)

    def action_notes(
        self, assessment_id: str, *, limit: int = 100, offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Read only this assessment's notes; never create a file, table, or journal."""
        assessment_id = _text(assessment_id, "原判断标识", 100)
        if (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
                or isinstance(offset, bool) or not isinstance(offset, int) or offset < 0):
            raise ValidationError("笔记条数必须为正整数，起点必须为非负整数")
        if not self.path.exists():
            return []
        with self._connection("ro") as conn:
            self._check_database(conn)
            if not self._has_action_notes(conn):
                return []
            rows = conn.execute(
                "SELECT * FROM assessment_action_notes WHERE assessment_id=? "
                "ORDER BY recorded_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (assessment_id, limit, offset),
            ).fetchall()
            return [self._action_note_record(row) for row in rows]


def _valid_nav(nav: Mapping[str, Any] | None) -> tuple[Decimal, str, str] | None:
    if not isinstance(nav, Mapping):
        return None
    try:
        value = decimal(nav.get("value"))
        currency = nav.get("currency")
        nav_date = nav.get("date")
        if (value <= 0 or not isinstance(currency, str)
                or not re.fullmatch(r"[A-Z]{3}", currency)
                or not isinstance(nav_date, str)
                or date.fromisoformat(nav_date).isoformat() != nav_date):
            return None
        return value, currency, nav_date
    except (ValidationError, ValueError, TypeError):
        return None


def value_holding(fund: Mapping[str, Any], nav: Mapping[str, Any] | None) -> dict[str, Any]:
    """Value confirmed shares against a positive, dated CNY unit NAV.

    Unknown quantities remain None. Foreign NAVs are not converted or subtracted
    from CNY cost. ``ready`` describes the market value, not cost completeness.
    ``profit_rate`` is a decimal ratio, not a percentage (0.1 means 10%).
    """
    shares = _holding_text(fund.get("shares"), "份额")
    cost = _holding_text(fund.get("cost"), "持仓总成本")
    currency = fund.get("currency", "CNY")
    parsed_nav = _valid_nav(nav)
    result = {
        "shares": shares, "cost": cost, "currency": currency,
        "market_value": None, "profit": None, "profit_rate": None,
        "nav_date": parsed_nav[2] if parsed_nav else None,
        "valuation_status": "ready",
    }
    if shares is None:
        result["valuation_status"] = "missing_shares"
    elif parsed_nav is None:
        result["valuation_status"] = "missing_nav"
    elif currency != "CNY" or parsed_nav[1] != "CNY":
        result["valuation_status"] = "currency_mismatch"
    else:
        # domain.decimal bounds each input to <=49 significant digits. Avoid
        # rounding their product under Decimal's default 28-digit context.
        with localcontext() as context:
            context.prec = 128
            market_value = decimal(shares) * parsed_nav[0]
            result["market_value"] = str(market_value)
            if cost is not None:
                basis = decimal(cost)
                profit = market_value - basis
                result["profit"] = str(profit)
                if basis > 0:
                    result["profit_rate"] = str(profit / basis)
    return result


def _valued_amount(value: Any) -> Decimal | None:
    """Derived market values can exceed domain.decimal's input magnitude limit."""
    if value is None or value == "":
        return None
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError("估值汇总需要有效十进制数") from exc
    if not result.is_finite() or isinstance(value, bool):
        raise ValidationError("估值汇总需要有限十进制数")
    return result


def summary_holdings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum valued rows in CNY, retaining each number's actual known cohort.

    Cost is independently known even without a NAV. Profit requires both market
    value and cost on the same row. The return ratio uses only those matched rows
    with positive cost (both numerator and denominator), excluding zero-cost rows.
    Empty cohorts return None, not a fabricated zero.
    """
    markets: list[Decimal] = []
    costs: list[Decimal] = []
    profits: list[Decimal] = []
    rate_profits: list[Decimal] = []
    rate_costs: list[Decimal] = []
    with localcontext() as context:
        context.prec = 128 + len(str(len(rows)))
        for row in rows:
            if row.get("currency") != "CNY":
                continue
            market = _valued_amount(row.get("market_value"))
            cost = _valued_amount(row.get("cost"))
            if market is not None:
                markets.append(market)
            if cost is not None:
                costs.append(cost)
            if market is not None and cost is not None:
                profit = market - cost
                profits.append(profit)
                if cost > 0:
                    rate_profits.append(profit)
                    rate_costs.append(cost)
        return {
            "currency": "CNY",
            "market_value": str(sum(markets, Decimal(0))) if markets else None,
            "cost": str(sum(costs, Decimal(0))) if costs else None,
            "profit": str(sum(profits, Decimal(0))) if profits else None,
            "profit_rate": str(sum(rate_profits, Decimal(0)) / sum(rate_costs, Decimal(0)))
            if rate_costs else None,
            "market_value_count": len(markets), "cost_count": len(costs),
            "profit_count": len(profits), "total_count": len(rows),
        }
