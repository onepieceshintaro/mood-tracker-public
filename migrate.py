"""ローカルSQLite（mood-tracker/mood.db）→ Supabase Postgres への一回限りの移行スクリプト。

使い方：
  1. .streamlit/secrets.toml に DATABASE_URL（Supabaseの接続文字列）を設定
     もしくは環境変数 DATABASE_URL をセット
  2. python migrate.py
     （--source で別パスのSQLiteを指定可能。デフォルトは ../mood-tracker/mood.db）

安全性：
  - 既存レコードは ON CONFLICT で上書き（user_id + log_date の複合キー）
  - 何度流しても結果は同じ（冪等）
"""
import argparse
import sqlite3
import sys
from pathlib import Path

# Windows の cp932 コンソールでも絵文字付きログが落ちないように UTF-8 へ
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlalchemy import text

from db import get_engine, init_db, is_postgres


DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "mood-tracker" / "mood.db"


COLUMNS = [
    "user_id",
    "log_date",
    "mood",
    "sleep_hours",
    "energy",
    "note",
    "tags",
    "temperature",
    "weather_code",
    "precipitation",
    "pressure",
    "wake_time",
]


def _fetch_rows(sqlite_path: Path) -> list[dict]:
    if not sqlite_path.exists():
        print(f"❌ SQLite ファイルが見つかりません: {sqlite_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute("PRAGMA table_info(mood_logs)")
        existing_cols = {r[1] for r in cur.fetchall()}
        if not existing_cols:
            print("❌ mood_logs テーブルが存在しません。")
            sys.exit(1)

        select_cols = []
        for c in COLUMNS:
            if c in existing_cols:
                select_cols.append(c)
            elif c == "user_id":
                # 旧スキーマ（user_id なし）→ デフォルトの所有者IDにマップ
                select_cols.append("NULL AS user_id")
            else:
                select_cols.append(f"NULL AS {c}")

        sql = f"SELECT {', '.join(select_cols)} FROM mood_logs"
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        return rows
    finally:
        conn.close()


def _resolve_default_user_id() -> str:
    """user_id 列がない旧データ用のフォールバック。_user.py のローカル保存を使う。"""
    try:
        from _user import get_or_create_user_id
        return get_or_create_user_id()
    except Exception:
        # Streamlit 依存で失敗した場合はホームディレクトリの保存を直接読む
        p = Path.home() / ".note_apps_user_id"
        if p.exists():
            uid = p.read_text(encoding="utf-8").strip()
            if len(uid) == 32:
                return uid
        print("❌ user_id を解決できません。_user.py の保存ファイルが必要です。")
        sys.exit(1)


def migrate(sqlite_path: Path, dry_run: bool = False) -> None:
    rows = _fetch_rows(sqlite_path)
    if not rows:
        print("ℹ️  移行するレコードがありません。")
        return

    default_uid = None
    for r in rows:
        if not r.get("user_id"):
            if default_uid is None:
                default_uid = _resolve_default_user_id()
                print(f"ℹ️  user_id 未設定の行にフォールバックIDを適用: {default_uid[:8]}…")
            r["user_id"] = default_uid

    print(f"📦 {len(rows)} 件を移行します（source: {sqlite_path}）")
    if dry_run:
        print("  --dry-run 指定のため、実際の書き込みは行いません。")
        for r in rows[:3]:
            print(f"   sample: {r}")
        return

    engine = get_engine()
    # URL.render_as_string(hide_password=True) はSQLAlchemy 1.4/2.0両対応
    try:
        safe_url = engine.url.render_as_string(hide_password=True)
    except Exception:
        safe_url = f"{engine.url.drivername}://***@{engine.url.host}:{engine.url.port}/{engine.url.database}"
    print(f"🎯 target: {safe_url}")
    if not is_postgres():
        print("⚠️  ターゲットが Postgres ではありません。DATABASE_URL を確認してください。")

    init_db()

    placeholders = ", ".join(f":{c}" for c in COLUMNS)
    col_list = ", ".join(COLUMNS)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in COLUMNS if c not in ("user_id", "log_date")
    )
    upsert_sql = text(f"""
        INSERT INTO mood_logs ({col_list})
        VALUES ({placeholders})
        ON CONFLICT (user_id, log_date) DO UPDATE SET
            {update_set}
    """)

    with engine.begin() as conn:
        for r in rows:
            params = {c: r.get(c) for c in COLUMNS}
            conn.execute(upsert_sql, params)

    print(f"✅ 完了: {len(rows)} 件を upsert しました。")


def main() -> None:
    ap = argparse.ArgumentParser(description="SQLite → Supabase 移行")
    ap.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"移行元SQLiteファイル（既定: {DEFAULT_SOURCE}）",
    )
    ap.add_argument("--dry-run", action="store_true", help="書き込みせず件数だけ確認")
    args = ap.parse_args()
    migrate(args.source, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
