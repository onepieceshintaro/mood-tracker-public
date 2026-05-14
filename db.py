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

        # mood_predictions（翌日予測値の保存・答え合わせ用）
        # 「予測値を入力前に見せると anchoring bias」問題を回避するため、
        # 予測値は入力時には出さず、対応する実測が入った後にだけ答え合わせとして見せる。
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS mood_predictions (
                user_id TEXT NOT NULL,
                predicted_for_date TEXT NOT NULL,
                predicted_value {"DOUBLE PRECISION" if pg else "REAL"} NOT NULL,
                based_on_date TEXT,
                generated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, predicted_for_date)
            )
        """))


# ------------------- mood_predictions helpers -------------------
def save_prediction(
    user_id: str,
    predicted_for_date,
    predicted_value: float,
    based_on_date=None,
) -> None:
    """翌日予測を保存（同一日付は上書き）。"""
    from time_utils import now_jst
    sql = text("""
        INSERT INTO mood_predictions
        (user_id, predicted_for_date, predicted_value, based_on_date, generated_at)
        VALUES
        (:user_id, :predicted_for_date, :predicted_value, :based_on_date, :generated_at)
        ON CONFLICT (user_id, predicted_for_date) DO UPDATE SET
            predicted_value = EXCLUDED.predicted_value,
            based_on_date = EXCLUDED.based_on_date,
            generated_at = EXCLUDED.generated_at
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "user_id": user_id,
            "predicted_for_date": str(predicted_for_date),
            "predicted_value": float(predicted_value),
            "based_on_date": str(based_on_date) if based_on_date else None,
            "generated_at": now_jst().isoformat(),
        })


def get_prediction_for_date(user_id: str, target_date) -> dict | None:
    """指定日付に対して保存されている予測値を取得（無ければ None）。"""
    sql = text("""
        SELECT predicted_value, based_on_date, generated_at
        FROM mood_predictions
        WHERE user_id = :user_id AND predicted_for_date = :predicted_for_date
    """)
    with get_engine().connect() as conn:
        row = conn.execute(sql, {
            "user_id": user_id,
            "predicted_for_date": str(target_date),
        }).fetchone()
    if not row:
        return None
    return {
        "predicted_value": float(row[0]),
        "based_on_date": row[1],
        "generated_at": row[2],
    }
