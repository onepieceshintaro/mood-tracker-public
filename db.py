"""DB抽象レイヤー。ローカルSQLite / クラウドPostgres（Supabase）を切り替える。

優先順位：
  1. st.secrets["DATABASE_URL"]  （Streamlit Cloud）
  2. 環境変数 DATABASE_URL         （ローカル .env など）
  3. sqlite:///mood.db             （フォールバック・既存のローカルDB）
"""
import os
from functools import lru_cache

import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def _get_database_url() -> str:
    # 1. Streamlit Cloud secrets
    try:
        url = st.secrets.get("DATABASE_URL")
        if url:
            return url
    except Exception:
        pass
    # 2. 環境変数
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # 3. ローカルSQLite（フォールバック）
    return "sqlite:///mood.db"


def _normalize_url(url: str) -> str:
    """SQLAlchemy が認識できる形式に正規化。"""
    # Supabase のコピー値が postgres:// から始まる場合に対応
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://") and "+psycopg2" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    url = _normalize_url(_get_database_url())
    # pool_pre_ping: Supabaseの接続が切れていても自動再接続
    return create_engine(url, pool_pre_ping=True, future=True)


def is_postgres() -> bool:
    return "postgresql" in str(get_engine().url)


# ------------------- DDL -------------------
def init_db() -> None:
    """テーブルを作成（冪等）。SQLite/Postgres両方で動く構文を使う。"""
    engine = get_engine()
    pg = is_postgres()

    with engine.begin() as conn:
        # mood_logs
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS mood_logs (
                user_id TEXT NOT NULL,
                log_date TEXT NOT NULL,
                mood INTEGER NOT NULL,
                sleep_hours {"DOUBLE PRECISION" if pg else "REAL"},
                energy INTEGER,
                note TEXT,
                tags TEXT,
                recovery TEXT,
                temperature {"DOUBLE PRECISION" if pg else "REAL"},
                weather_code INTEGER,
                precipitation {"DOUBLE PRECISION" if pg else "REAL"},
                pressure {"DOUBLE PRECISION" if pg else "REAL"},
                wake_time TEXT,
                PRIMARY KEY (user_id, log_date)
            )
        """))
        # 既存テーブルに recovery が無い場合に後付けで追加（冪等）
        try:
            conn.execute(text(
                "ALTER TABLE mood_logs ADD COLUMN IF NOT EXISTS recovery TEXT"
            ))
        except Exception:
            pass
        # 2026-05-06: 睡眠の質（3択）とイベント変数を追加（既存データ保持・破壊なし）
        try:
            conn.execute(text(
                "ALTER TABLE mood_logs ADD COLUMN IF NOT EXISTS sleep_quality TEXT"
            ))
        except Exception:
            pass
        try:
            conn.execute(text(
                "ALTER TABLE mood_logs ADD COLUMN IF NOT EXISTS events TEXT"
            ))
        except Exception:
            pass

        # user_nicknames（3アプリ共通・プレフィックス無し）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_nicknames (
                user_id TEXT PRIMARY KEY,
                nickname TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """))

        # user_preferences（受動的な気づきメモ等のオプトイン設定）
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                notify_mood_dip {"BOOLEAN" if pg else "INTEGER"} NOT NULL DEFAULT {"FALSE" if pg else "0"},
                updated_at TEXT
            )
        """))
