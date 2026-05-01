"""ユーザーごとのオプトイン設定（受動的な気づきメモ等）。

思想：
- すべてデフォルトOFF（押しつけない）
- 本人がONにしたときだけ表示する
- 表示は事実のみ・断定や行動要求はしない
"""
from datetime import datetime

from sqlalchemy import text

from db import get_engine


def get_notify_mood_dip(user_id: str) -> bool:
    """気分の落ち込み傾向の気づきをONにしているか。デフォルトFalse。"""
    if not user_id:
        return False
    sql = text(
        "SELECT notify_mood_dip FROM user_preferences WHERE user_id = :uid"
    )
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"uid": user_id}).first()
    if row is None:
        return False
    val = row[0]
    # SQLiteは0/1, PostgresはBoolean
    return bool(val)


def set_notify_mood_dip(user_id: str, value: bool) -> None:
    """気分の落ち込み傾向の気づきをON/OFF。upsert。"""
    if not user_id:
        return
    now = datetime.now().isoformat()
    # PostgreSQL の ON CONFLICT を使う（SQLite 3.24+ も同構文で動く）
    sql = text("""
        INSERT INTO user_preferences (user_id, notify_mood_dip, updated_at)
        VALUES (:uid, :val, :now)
        ON CONFLICT (user_id) DO UPDATE SET
            notify_mood_dip = EXCLUDED.notify_mood_dip,
            updated_at = EXCLUDED.updated_at
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {"uid": user_id, "val": bool(value), "now": now})
