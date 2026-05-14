"""ユーザーごとのオプトイン設定（受動的な気づきメモ等）。

思想：
- すべてデフォルトOFF（押しつけない）
- 本人がONにしたときだけ表示する
- 表示は事実のみ・断定や行動要求はしない
- DB側のテーブル不在等で失敗しても、機能停止せずデフォルト（OFF）で動く
"""
from datetime import datetime

from sqlalchemy import text

from db import get_engine, is_postgres
from time_utils import now_jst_naive


def _ensure_table() -> None:
    """user_preferences が無ければ作る。冪等で軽量。
    既存テーブルに my_actions カラムが無ければ追加（冪等な ALTER）。
    """
    pg = is_postgres()
    bool_type = "BOOLEAN" if pg else "INTEGER"
    default_val = "FALSE" if pg else "0"
    sql = text(f"""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            notify_mood_dip {bool_type} NOT NULL DEFAULT {default_val},
            my_actions TEXT,
            updated_at TEXT
        )
    """)
    with get_engine().begin() as conn:
        conn.execute(sql)
        # 既存テーブル（my_actions カラム無し）への後付け
        try:
            conn.execute(text(
                "ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS my_actions TEXT"
            ))
        except Exception:
            # SQLite は IF NOT EXISTS をサポートしないので別途試す
            try:
                conn.execute(text(
                    "ALTER TABLE user_preferences ADD COLUMN my_actions TEXT"
                ))
            except Exception:
                # すでにある等 → 無視
                pass


def get_notify_mood_dip(user_id: str) -> bool:
    """気分の落ち込み傾向の気づきをONにしているか。デフォルトFalse。"""
    if not user_id:
        return False
    sql = text(
        "SELECT notify_mood_dip FROM user_preferences WHERE user_id = :uid"
    )
    try:
        with get_engine().connect() as conn:
            row = conn.execute(sql, {"uid": user_id}).first()
        if row is None:
            return False
        # SQLiteは0/1, PostgresはBoolean
        return bool(row[0])
    except Exception:
        # テーブル未作成等の場合：作成を試みて、結果はFalse
        try:
            _ensure_table()
        except Exception:
            pass
        return False


def set_notify_mood_dip(user_id: str, value: bool) -> None:
    """気分の落ち込み傾向の気づきをON/OFF。upsert。失敗時は何もしない。"""
    if not user_id:
        return
    # 冪等にテーブルを確保（既にあれば即終わる）
    try:
        _ensure_table()
    except Exception:
        # テーブル作成自体に失敗（権限不足等）したら、設定保存は諦める
        return

    now = now_jst_naive().isoformat()
    sql = text("""
        INSERT INTO user_preferences (user_id, notify_mood_dip, updated_at)
        VALUES (:uid, :val, :now)
        ON CONFLICT (user_id) DO UPDATE SET
            notify_mood_dip = EXCLUDED.notify_mood_dip,
            updated_at = EXCLUDED.updated_at
    """)
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, {"uid": user_id, "val": bool(value), "now": now})
    except Exception:
        # フェイルセーフ：書き込み失敗してもアプリは止めない
        pass


def get_my_actions(user_id: str) -> str:
    """ユーザーが自分で書いた『試したい選択肢メモ』を取得。デフォルト空文字。"""
    if not user_id:
        return ""
    sql = text("SELECT my_actions FROM user_preferences WHERE user_id = :uid")
    try:
        with get_engine().connect() as conn:
            row = conn.execute(sql, {"uid": user_id}).first()
        if row is None or row[0] is None:
            return ""
        return str(row[0])
    except Exception:
        try:
            _ensure_table()
        except Exception:
            pass
        return ""


def set_my_actions(user_id: str, content: str) -> None:
    """『試したい選択肢メモ』を保存。upsert。失敗時は何もしない。"""
    if not user_id:
        return
    try:
        _ensure_table()
    except Exception:
        return
    now = now_jst_naive().isoformat()
    sql = text("""
        INSERT INTO user_preferences (user_id, my_actions, updated_at)
        VALUES (:uid, :val, :now)
        ON CONFLICT (user_id) DO UPDATE SET
            my_actions = EXCLUDED.my_actions,
            updated_at = EXCLUDED.updated_at
    """)
    try:
        with get_engine().begin() as conn:
            conn.execute(sql, {"uid": user_id, "val": content or "", "now": now})
    except Exception:
        pass
