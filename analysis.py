"""気分データの分析：曜日別分析、翌日の気分との相関、予測モデル、観察ベースの気づき"""
import pandas as pd
import numpy as np


# =========================
# 観察ベースの気づき（事実のみ・解釈しない）
# =========================
def streak_days(df: pd.DataFrame) -> int:
    """今日まで何日連続で記録できているか（連続日数）。"""
    if df is None or df.empty:
        return 0
    today = pd.Timestamp.now().normalize()
    dates = pd.to_datetime(df["log_date"]).dt.normalize().sort_values().unique()
    if len(dates) == 0:
        return 0
    streak = 0
    expected = today
    for d in reversed(dates):
        if d == expected or d == expected - pd.Timedelta(days=1):
            streak += 1
            expected = d - pd.Timedelta(days=1)
        else:
            break
    return streak


def daily_observations(df: pd.DataFrame) -> list[str]:
    """蓄積データから「事実」だけを返す。解釈や予測はしない。

    返すパターン例:
    - 連続記録日数
    - 直近7日の平均と前7日の比較
    - 直近の睡眠の傾向（短い日が続いているか等）
    - 直近の天気・気圧の特徴
    - 出来事タグの頻度
    """
    if df is None or df.empty:
        return []
    out: list[str] = []
    df = df.copy()
    df["log_date"] = pd.to_datetime(df["log_date"]).dt.normalize()
    df = df.sort_values("log_date").reset_index(drop=True)
    today = pd.Timestamp.now().normalize()

    # 1) 連続記録日数
    s = streak_days(df)
    if s >= 2:
        out.append(f"🌱 **{s}日連続**で記録できています。")
    elif s == 1:
        out.append("🌱 今日も記録できました。記録1日目から始まります。")

    # 2) 直近7日 vs その前7日 の気分平均
    cutoff_recent = today - pd.Timedelta(days=7)
    cutoff_prev = today - pd.Timedelta(days=14)
    recent7 = df[df["log_date"] > cutoff_recent]
    prev7 = df[(df["log_date"] <= cutoff_recent) & (df["log_date"] > cutoff_prev)]
    if len(recent7) >= 3 and len(prev7) >= 3:
        m_recent = recent7["mood"].mean()
        m_prev = prev7["mood"].mean()
        diff = m_recent - m_prev
        if abs(diff) >= 0.5:
            direction = "上がっています" if diff > 0 else "下がっています"
            out.append(
                f"📊 直近7日の気分平均は **{m_recent:.1f}**、"
                f"その前7日（{m_prev:.1f}）より **{abs(diff):.1f} {direction}**。"
            )
        elif len(recent7) >= 5:
            out.append(
                f"📊 直近7日の気分平均は **{m_recent:.1f}**、ほぼ前週と変わりません。"
            )

    # 3) 直近7日の睡眠（時間がある日のみ）
    if "sleep_hours" in df.columns:
        sh_recent = recent7["sleep_hours"].dropna()
        if len(sh_recent) >= 3:
            short_days = int((sh_recent < 6).sum())
            if short_days >= 3:
                out.append(
                    f"😴 直近7日のうち **{short_days}日** で睡眠が6時間未満でした。"
                )

    # 4) 気圧の急変
    if "pressure" in df.columns and len(recent7) >= 3:
        pres = recent7["pressure"].dropna()
        if len(pres) >= 3:
            pres_diff = pres.diff().dropna()
            big_drops = int((pres_diff < -8).sum())
            if big_drops >= 1:
                out.append(
                    f"🌡 直近7日で気圧が急に下がった日が **{big_drops}日** ありました。"
                )

    # 5) 出来事タグの頻度（直近30日）
    cutoff_30 = today - pd.Timedelta(days=30)
    last30 = df[df["log_date"] >= cutoff_30]
    if "tags" in last30.columns and len(last30) >= 7:
        from collections import Counter
        tag_counter: Counter = Counter()
        for s in last30["tags"].dropna():
            for t in str(s).split(","):
                t = t.strip()
                if t:
                    tag_counter[t] += 1
        if tag_counter:
            top1, c1 = tag_counter.most_common(1)[0]
            out.append(
                f"🏷 直近30日でいちばん多かった出来事タグは **「{top1}」**（{c1}日分）。"
            )

    # 6) 「良かったこと」を書いた日数
    if "recovery" in last30.columns:
        rec_count = int(
            last30["recovery"].fillna("").astype(str).str.len().gt(0).sum()
        )
        if rec_count >= 3:
            out.append(
                f"✨ 直近30日で **{rec_count}日**、「良かったこと」を書きました。"
            )

    return out

JP_DOW = {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
DOW_ORDER = ["月", "火", "水", "木", "金", "土", "日"]

# 予測対象（翌日の気分）との相関を調べる候補特徴量
BASE_FEATURES = [
    "mood", "sleep_hours", "energy",
    "temperature", "pressure", "precipitation",
    "wake_minutes",
]
DERIVED_FEATURES = [
    "pressure_delta", "temp_delta", "sleep_delta",
    "wake_delta", "wake_ma7_gap",
    "mood_ma3", "mood_ma7",
]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES

FEATURE_JP = {
    "mood": "当日の気分",
    "sleep_hours": "睡眠時間",
    "energy": "エネルギー",
    "temperature": "気温",
    "pressure": "気圧",
    "precipitation": "降水量",
    "wake_minutes": "起床時刻",
    "pressure_delta": "気圧の前日差",
    "temp_delta": "気温の前日差",
    "sleep_delta": "睡眠の前日差",
    "wake_delta": "起床の前日差",
    "wake_ma7_gap": "起床の7日平均からの乖離",
    "mood_ma3": "直近3日の気分平均",
    "mood_ma7": "直近7日の気分平均",
}


def _wake_to_minutes(s):
    """'HH:MM' → 分換算（深夜0時基準）。Noneや不正値はNaN。"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    try:
        h, m = map(int, str(s).split(":"))
        return h * 60 + m
    except Exception:
        return np.nan


def dow_stats(df: pd.DataFrame) -> pd.DataFrame:
    """曜日別の気分統計（平均・標準偏差・件数）"""
    df = df.copy()
    df["log_date"] = pd.to_datetime(df["log_date"])
    df["曜日"] = df["log_date"].dt.dayofweek.map(JP_DOW)

    stats = df.groupby("曜日")["mood"].agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["曜日", "平均気分", "標準偏差", "記録数"]
    stats["曜日"] = pd.Categorical(stats["曜日"], categories=DOW_ORDER, ordered=True)
    stats = stats.sort_values("曜日").reset_index(drop=True)
    stats["平均気分"] = stats["平均気分"].round(2)
    stats["標準偏差"] = stats["標準偏差"].round(2)
    return stats


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """ML用の特徴量フレーム。前日差・移動平均・翌日気分（target）を付与。"""
    d = df.copy()
    d["log_date"] = pd.to_datetime(d["log_date"])
    d = d.sort_values("log_date").set_index("log_date")

    # 日付の欠損を埋めて連続化（欠損日はNaN→後で除外）
    if len(d) > 0:
        idx = pd.date_range(d.index.min(), d.index.max(), freq="D")
        d = d.reindex(idx)

    # 起床時刻 → 分換算
    if "wake_time" in d.columns:
        d["wake_minutes"] = d["wake_time"].apply(_wake_to_minutes)

    # 前日差
    for col in ["pressure", "temperature", "sleep_hours"]:
        if col in d.columns:
            d[f"{col.split('_')[0]}_delta" if col != "sleep_hours" else "sleep_delta"] = d[col].diff()

    # 起床の前日差・7日平均との乖離
    if "wake_minutes" in d.columns:
        d["wake_delta"] = d["wake_minutes"].diff()
        wake_ma7 = d["wake_minutes"].rolling(7, min_periods=3).mean()
        d["wake_ma7_gap"] = d["wake_minutes"] - wake_ma7

    # 移動平均
    d["mood_ma3"] = d["mood"].rolling(3, min_periods=1).mean()
    d["mood_ma7"] = d["mood"].rolling(7, min_periods=1).mean()

    # 曜日
    d["dow"] = d.index.dayofweek

    # 目的変数：翌日の気分
    d["mood_next"] = d["mood"].shift(-1)

    return d.reset_index().rename(columns={"index": "log_date"})


def correlations_with_next_mood(df: pd.DataFrame) -> pd.DataFrame:
    """各特徴量と『翌日の気分』とのPearson相関"""
    feat = build_feature_frame(df)
    results = []
    for f in ALL_FEATURES:
        if f not in feat.columns:
            continue
        sub = feat[[f, "mood_next"]].dropna()
        if len(sub) < 5:
            continue
        corr = sub[f].corr(sub["mood_next"])
        if pd.isna(corr):
            continue
        results.append({
            "特徴量": FEATURE_JP.get(f, f),
            "相関係数": round(corr, 3),
            "サンプル数": len(sub),
        })
    if not results:
        return pd.DataFrame(columns=["特徴量", "相関係数", "サンプル数"])
    return pd.DataFrame(results).sort_values(
        "相関係数", key=lambda s: s.abs(), ascending=False
    ).reset_index(drop=True)


def train_mood_predictor(df: pd.DataFrame, min_samples: int = 14):
    """線形回帰で翌日の気分を予測。データ不足時は Noneを返す。
    戻り値 dict:
      - n_train: 学習に使えたサンプル数
      - train_r2: 学習データへのR²（過学習評価）
      - cv_r2: 5分割CVのR²平均（実質的な予測力）
      - importance: 標準化係数（重要度 DataFrame）
      - in_sample_predictions: 学習データへの当てはめ（参考）
      - cv_predictions: 各日を一度学習から外して予測した値（実質的な予測 vs 実測）
      - next_day: 「明日の予測」 dict {"date", "predicted_mood", "based_on"}
      - features_used: 使用した特徴量リスト
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score, cross_val_predict

    feat = build_feature_frame(df)
    features = [f for f in ALL_FEATURES if f in feat.columns]
    train = feat.dropna(subset=["mood_next"] + features)

    if len(train) < min_samples:
        return {"n_train": len(train), "required": min_samples}

    X = train[features]
    y = train["mood_next"]

    model = LinearRegression().fit(X, y)

    # CV R²と CV予測（サンプル少ない時は無理しない）
    cv_n = min(5, max(2, len(train) // 3))
    try:
        cv_r2 = cross_val_score(model, X, y, cv=cv_n, scoring="r2").mean()
    except Exception:
        cv_r2 = None
    try:
        cv_y_pred = cross_val_predict(LinearRegression(), X, y, cv=cv_n)
        cv_predictions = pd.DataFrame({
            "日付": train["log_date"].reset_index(drop=True),
            "実測": y.reset_index(drop=True),
            "予測（CV・未学習相当）": cv_y_pred,
        })
    except Exception:
        cv_predictions = None

    # 標準化係数（特徴量の寄与度）
    X_std = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    std_model = LinearRegression().fit(X_std, y)
    importance = pd.DataFrame({
        "特徴量": [FEATURE_JP.get(f, f) for f in features],
        "効き方": std_model.coef_.round(3),
    })
    importance["寄与度（絶対値）"] = importance["効き方"].abs().round(3)
    importance = importance.sort_values("寄与度（絶対値）", ascending=False).reset_index(drop=True)

    in_sample_predictions = pd.DataFrame({
        "日付": train["log_date"].reset_index(drop=True),
        "実測": y.reset_index(drop=True),
        "当てはめ": model.predict(X),
    })

    # 「明日の予測」：特徴量がそろっている最新の行から、翌日の気分を予測
    next_day = None
    feat_ready = feat.dropna(subset=features)
    if len(feat_ready) > 0:
        last_row = feat_ready.iloc[-1]
        try:
            raw_pred = float(model.predict([last_row[features].values])[0])
            clamped = max(1.0, min(10.0, raw_pred))
            base_date = pd.to_datetime(last_row["log_date"]).date()
            next_day = {
                "date": (pd.to_datetime(last_row["log_date"]) + pd.Timedelta(days=1)).date(),
                "based_on_date": base_date,
                "predicted_mood": round(clamped, 1),
                "raw_prediction": round(raw_pred, 2),
                "clamped": raw_pred != clamped,
            }
        except Exception:
            next_day = None

    return {
        "n_train": len(train),
        "train_r2": round(model.score(X, y), 3),
        "cv_r2": round(cv_r2, 3) if cv_r2 is not None else None,
        "importance": importance,
        "in_sample_predictions": in_sample_predictions,
        "cv_predictions": cv_predictions,
        "next_day": next_day,
        "features_used": [FEATURE_JP.get(f, f) for f in features],
    }
