"""
気分トラッキング（クラウド対応版）
- DB：ローカルSQLite / Supabase Postgres 自動切替（db.py）
- 天気・気圧の自動取得（Open-Meteo）
- 曜日別分析
- 翌日の気分予測（線形回帰）
"""
from datetime import date, time, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import text

from weather import (
    fetch_weather, describe_weather, geocode_city,
    DEFAULT_LAT, DEFAULT_LON,
)
from analysis import (
    dow_stats, correlations_with_next_mood, train_mood_predictor, DOW_ORDER,
)
from _user import render_account_sidebar, get_or_create_user_id
from db import get_engine, init_db


# -------------- タグ定義 --------------
TAG_OPTIONS = ["仕事", "人間関係", "家族", "通院・治療", "休息", "外出", "運動"]
TAG_EMOJI = {
    "仕事": "🏢",
    "人間関係": "👥",
    "家族": "👨‍👩‍👧",
    "通院・治療": "🏥",
    "休息": "🛋",
    "外出": "🚶",
    "運動": "🏃",
}


def _format_tag(t: str) -> str:
    return f"{TAG_EMOJI.get(t, '')} {t}".strip()


def _parse_tag_string(raw: str) -> list[str]:
    """保存された tags 文字列をリストに戻す（全角読点・カンマ両対応）。"""
    if not raw:
        return []
    normalized = raw.replace("、", ",")
    return [t.strip() for t in normalized.split(",") if t.strip()]


# -------------- 傾向サマリ（本人データからの事実のみ） --------------
def build_insights(df: pd.DataFrame) -> list[str]:
    """本人の記録から事実ベースの短文を返す。

    原則：
      - 一般論は返さない（「睡眠は大事」的な言い回しNG）
      - 全て本人データから計算した数値のみ
      - サンプル数が少ないものは除外（最低 n=3）
      - 全体平均との差が小さいものは除外（|diff| < 0.5）
    """
    out: list[str] = []
    if df is None or len(df) < 14:
        return out

    overall = df["mood"].mean()

    # 1. 睡眠6h未満の翌日
    if "sleep_hours" in df.columns:
        d = df.sort_values("log_date").reset_index(drop=True).copy()
        d["next_mood"] = d["mood"].shift(-1)
        nxt = d[d["sleep_hours"] < 6]["next_mood"].dropna()
        if len(nxt) >= 3:
            diff = nxt.mean() - overall
            if abs(diff) >= 0.5:
                direction = "低い" if diff < 0 else "高い"
                out.append(
                    f"💤 **睡眠6h未満**の翌日、気分平均は **{nxt.mean():.1f}** "
                    f"（全体平均より {abs(diff):.1f} {direction}・n={len(nxt)}）"
                )

    # 2. 曜日（最も低い曜日・最も高い曜日）
    d_dow = df.copy()
    d_dow["dow"] = d_dow["log_date"].dt.dayofweek
    dow_names = ["月", "火", "水", "木", "金", "土", "日"]
    stats = d_dow.groupby("dow")["mood"].agg(["mean", "count"])
    stats = stats[stats["count"] >= 3]
    if not stats.empty:
        worst_dow = stats["mean"].idxmin()
        worst_diff = stats.loc[worst_dow, "mean"] - overall
        if worst_diff <= -0.5:
            out.append(
                f"📅 **{dow_names[worst_dow]}曜日**の気分は全体平均より "
                f"**{abs(worst_diff):.1f}** 低い傾向 "
                f"（平均 {stats.loc[worst_dow, 'mean']:.1f}・"
                f"n={int(stats.loc[worst_dow, 'count'])}）"
            )
        best_dow = stats["mean"].idxmax()
        best_diff = stats.loc[best_dow, "mean"] - overall
        if best_diff >= 0.5 and best_dow != worst_dow:
            out.append(
                f"📅 **{dow_names[best_dow]}曜日**の気分は全体平均より "
                f"**{best_diff:.1f}** 高い傾向 "
                f"（平均 {stats.loc[best_dow, 'mean']:.1f}・"
                f"n={int(stats.loc[best_dow, 'count'])}）"
            )

    # 3. 低気圧の日
    if "pressure" in df.columns:
        dp = df[df["pressure"].notna()]
        if len(dp) >= 10:
            threshold = 1005
            low = dp[dp["pressure"] <= threshold]["mood"]
            if len(low) >= 3:
                diff = low.mean() - overall
                if abs(diff) >= 0.5:
                    direction = "低い" if diff < 0 else "高い"
                    out.append(
                        f"🌡 気圧 **{threshold}hPa以下**の日、気分平均 **{low.mean():.1f}** "
                        f"（全体平均より {abs(diff):.1f} {direction}・n={len(low)}）"
                    )

    # 4. タグ別（最も高い / 最も低い）
    if "tags" in df.columns:
        td = df[["mood", "tags"]].copy()
        td["tag_list"] = td["tags"].fillna("").apply(_parse_tag_string)
        records = []
        for t in TAG_OPTIONS:
            sub = td[td["tag_list"].apply(lambda lst: t in lst)]
            if len(sub) >= 3:
                records.append((t, sub["mood"].mean(), len(sub)))
        if records:
            records.sort(key=lambda x: x[1])
            worst_t, worst_m, worst_n = records[0]
            worst_diff = worst_m - overall
            if worst_diff <= -0.5:
                out.append(
                    f"{TAG_EMOJI.get(worst_t, '')} **「{worst_t}」**と書いた日の気分平均 "
                    f"**{worst_m:.1f}**（全体平均より {worst_diff:.1f}・n={worst_n}）"
                )
            best_t, best_m, best_n = records[-1]
            best_diff = best_m - overall
            if best_diff >= 0.5 and best_t != worst_t:
                out.append(
                    f"{TAG_EMOJI.get(best_t, '')} **「{best_t}」**と書いた日の気分平均 "
                    f"**{best_m:.1f}**（全体平均より +{best_diff:.1f}・n={best_n}）"
                )

    return out


# -------------- DB 操作 --------------
def upsert(log_date, mood, sleep_hours, energy, note, tags, weather, wake_time,
           recovery="", user_id=None):
    if user_id is None:
        user_id = get_or_create_user_id()
    w = weather or {}
    wake_str = wake_time.strftime("%H:%M") if wake_time else None
    sql = text("""
        INSERT INTO mood_logs
        (user_id, log_date, mood, sleep_hours, energy, note, tags, recovery,
         temperature, weather_code, precipitation, pressure, wake_time)
        VALUES
        (:user_id, :log_date, :mood, :sleep_hours, :energy, :note, :tags, :recovery,
         :temperature, :weather_code, :precipitation, :pressure, :wake_time)
        ON CONFLICT (user_id, log_date) DO UPDATE SET
            mood = EXCLUDED.mood,
            sleep_hours = EXCLUDED.sleep_hours,
            energy = EXCLUDED.energy,
            note = EXCLUDED.note,
            tags = EXCLUDED.tags,
            recovery = EXCLUDED.recovery,
            temperature = EXCLUDED.temperature,
            weather_code = EXCLUDED.weather_code,
            precipitation = EXCLUDED.precipitation,
            pressure = EXCLUDED.pressure,
            wake_time = EXCLUDED.wake_time
    """)
    with get_engine().begin() as conn:
        conn.execute(sql, {
            "user_id": user_id,
            "log_date": str(log_date),
            "mood": mood,
            "sleep_hours": sleep_hours,
            "energy": energy,
            "note": note,
            "tags": tags,
            "recovery": recovery,
            "temperature": w.get("temperature"),
            "weather_code": w.get("weather_code"),
            "precipitation": w.get("precipitation"),
            "pressure": w.get("pressure"),
            "wake_time": wake_str,
        })


def load_existing(log_date, user_id=None):
    if user_id is None:
        user_id = get_or_create_user_id()
    sql = text("""
        SELECT mood, sleep_hours, energy, note, tags, recovery,
               temperature, weather_code, precipitation, pressure, wake_time
        FROM mood_logs WHERE user_id = :user_id AND log_date = :log_date
    """)
    with get_engine().connect() as conn:
        row = conn.execute(
            sql, {"user_id": user_id, "log_date": str(log_date)}
        ).fetchone()
    return row


def latest_wake_time(user_id=None):
    if user_id is None:
        user_id = get_or_create_user_id()
    sql = text("""
        SELECT wake_time FROM mood_logs
        WHERE user_id = :user_id AND wake_time IS NOT NULL
        ORDER BY log_date DESC LIMIT 1
    """)
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"user_id": user_id}).fetchone()
    if row and row[0]:
        try:
            h, m = map(int, row[0].split(":"))
            return time(h, m)
        except Exception:
            return None
    return None


def load_all(user_id=None):
    if user_id is None:
        user_id = get_or_create_user_id()
    sql = text(
        "SELECT * FROM mood_logs WHERE user_id = :user_id ORDER BY log_date"
    )
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"user_id": user_id})
    return df


# -------------- UI --------------
st.set_page_config(page_title="気分の記録", page_icon="📝", layout="centered")
CURRENT_USER_ID = render_account_sidebar()
init_db()

# サイドバー：位置情報
if "location" not in st.session_state:
    st.session_state["location"] = {
        "lat": DEFAULT_LAT, "lon": DEFAULT_LON,
        "label": "兵庫県 明石市",
    }

with st.sidebar:
    _hub_url = "https://app-public-qpy8b2ziwgdf9h2vmu5hqp.streamlit.app/"
    if CURRENT_USER_ID:
        _hub_url += f"?u={CURRENT_USER_ID}"
    st.link_button(
        "🏠 HOME に戻る",
        _hub_url,
        use_container_width=True,
    )
    st.link_button(
        "💬 ご意見・感想",
        "https://docs.google.com/forms/d/e/1FAIpQLSetCb_dHG6JFsUzhK9ZYxydgh5cP8w07Q6NRO4ouEM7BvSTRw/viewform",
        use_container_width=True,
    )
    st.divider()
    st.header("位置設定")
    st.caption("天気データの取得に使います")

    city = st.text_input("市名・地名で検索", placeholder="例: 明石市、神戸、Osaka")
    st.caption("⚠️ 「新宿区」など**区名は非対応**（API制約）。"
               "親市区町村や近隣市で指定してください。")
    if st.button("🔍 この場所に設定", use_container_width=True):
        if city.strip():
            with st.spinner("場所を検索中..."):
                hit = geocode_city(city)
            if hit:
                st.session_state["location"] = {
                    "lat": hit["latitude"], "lon": hit["longitude"],
                    "label": " ".join(
                        [x for x in [hit.get("admin1"), hit.get("name")] if x]
                    ) or city,
                }
                for k in [k for k in st.session_state if k.startswith("weather_")]:
                    del st.session_state[k]
                st.rerun()
            else:
                st.error("見つかりませんでした。別の書き方で試してみてください。")

    loc = st.session_state["location"]
    st.success(f"📍 {loc['label']}")
    st.caption(f"（{loc['lat']:.4f}, {loc['lon']:.4f}）")

    with st.expander("緯度経度を直接入力"):
        lat_in = st.number_input("緯度", value=loc["lat"], format="%.4f", key="lat_in")
        lon_in = st.number_input("経度", value=loc["lon"], format="%.4f", key="lon_in")
        if st.button("この座標に設定", use_container_width=True):
            st.session_state["location"] = {
                "lat": lat_in, "lon": lon_in,
                "label": f"手動設定 ({lat_in:.2f}, {lon_in:.2f})",
            }
            for k in [k for k in st.session_state if k.startswith("weather_")]:
                del st.session_state[k]
            st.rerun()

lat = st.session_state["location"]["lat"]
lon = st.session_state["location"]["lon"]

st.markdown("### 📝 今日の気分")
st.caption("30秒で書ける、浅くて続けられる記録。")

# --- 日付と天気（フォームの外で、変更即時反映） ---
log_date = st.date_input("日付", value=date.today(), max_value=date.today())

weather_cache_key = f"weather_{log_date}_{lat}_{lon}"
col_w1, col_w2 = st.columns([4, 1])
with col_w1:
    if weather_cache_key not in st.session_state:
        with st.spinner("天気情報を取得中..."):
            st.session_state[weather_cache_key] = fetch_weather(log_date, lat, lon)
    weather = st.session_state[weather_cache_key]

    if weather:
        emoji, label = describe_weather(weather["weather_code"])
        pressure_str = f"{weather['pressure']:.0f} hPa" if weather["pressure"] else "—"
        precip_str = f"{weather['precipitation']:.1f}mm" if weather["precipitation"] else "0mm"
        st.info(
            f"{emoji} {label}｜"
            f"気温 **{weather['temperature']:.1f}°C**｜"
            f"気圧 **{pressure_str}**｜"
            f"降水 {precip_str}"
        )
    else:
        st.warning("天気情報を取得できませんでした（オフライン？）")
with col_w2:
    if st.button("🔄 再取得", use_container_width=True):
        if weather_cache_key in st.session_state:
            del st.session_state[weather_cache_key]
        st.rerun()

existing = load_existing(log_date)
if existing:
    st.caption(f"📌 {log_date} は既に記録があります。上書きもできます。")
    (init_mood, init_sleep, init_energy, init_note, init_tags, init_recovery,
     _t, _wc, _p, _pr, init_wake_str) = existing
else:
    init_mood, init_sleep, init_energy, init_note, init_tags, init_recovery = (
        5, 7.0, 5, "", "", ""
    )
    init_wake_str = None

if init_wake_str:
    try:
        h, m = map(int, init_wake_str.split(":"))
        init_wake = time(h, m)
    except Exception:
        init_wake = latest_wake_time() or time(7, 0)
else:
    init_wake = latest_wake_time() or time(7, 0)

with st.form("mood_form"):
    # --- 基本（必須・常に表示） ---
    col1, col2 = st.columns(2)
    with col1:
        mood = st.slider(
            "気分（1=とても悪い 〜 10=とても良い）", 1, 10, init_mood or 5
        )
    with col2:
        energy = st.slider(
            "エネルギー（1=枯渇 〜 10=元気いっぱい）", 1, 10, init_energy or 5
        )

    col3, col4 = st.columns(2)
    with col3:
        sleep_hours = st.number_input(
            "睡眠時間（h）", 0.0, 14.0, float(init_sleep or 7.0), 0.5
        )
    with col4:
        wake_time = st.time_input(
            "起床時刻", value=init_wake, step=300,
            help="いつもより遅く/早く起きた日がないか見るために使います",
        )

    # --- 任意（折りたたみ・書きたい日だけ） ---
    _init_tag_list = [t for t in _parse_tag_string(init_tags or "") if t in TAG_OPTIONS]
    _has_optional = bool(_init_tag_list or init_recovery or init_note)
    with st.expander("もう少し書く（任意）", expanded=_has_optional):
        tags_selected = st.multiselect(
            "出来事（複数選択可）",
            TAG_OPTIONS,
            default=_init_tag_list,
            format_func=_format_tag,
        )
        st.caption(
            "目安：**仕事**（業務・職場）／"
            "**人間関係**（家族以外：友人・同僚・知人など）／"
            "**家族**／**通院・治療**／**休息**（何もしない日・昼寝など）／"
            "**外出**（買い物・散歩・外食など）／**運動**"
        )
        recovery = st.text_input(
            "今日ちょっと良かったこと", value=init_recovery or "",
            placeholder="例: 散歩した、よく眠れた、友人と話した",
            help="書けない日はスキップで大丈夫です",
        )
        note = st.text_area(
            "一言メモ", value=init_note or "",
            placeholder="書きたい時だけで大丈夫です",
            height=80,
        )

    submitted = st.form_submit_button("記録する", use_container_width=True)
    if submitted:
        tags_to_save = ",".join(tags_selected)
        upsert(log_date, mood, sleep_hours, energy, note, tags_to_save,
               weather, wake_time, recovery=recovery)
        st.success(f"{log_date} の記録を保存しました")

st.divider()

# -------------- 可視化 --------------
st.header("傾向")

df = load_all()
if df.empty:
    st.info("まだ記録がありません。上のフォームから1件だけでも記録してみてください。")
    st.stop()

df["log_date"] = pd.to_datetime(df["log_date"])
df = df.sort_values("log_date").reset_index(drop=True)

# --- あなたの傾向（本人データからの事実） ---
_insights = build_insights(df)
if _insights:
    st.subheader("📊 あなたの傾向")
    st.caption("あなた自身の記録から出た事実です（一般論ではなく）")
    for s in _insights:
        st.markdown(f"- {s}")
    st.divider()
elif len(df) < 14:
    st.caption(
        f"📊 「あなたの傾向」は記録が14日分以上溜まると表示されます "
        f"（現在 {len(df)} 日）"
    )

# --- 最近の「良かったこと」ハイライト ---
if "recovery" in df.columns:
    _cutoff_rec = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    _rec_df = df[
        (df["log_date"] >= _cutoff_rec)
        & (df["recovery"].fillna("").astype(str).str.len() > 0)
    ][["log_date", "recovery"]]
    if not _rec_df.empty:
        st.subheader("✨ 最近の「良かったこと」")
        st.caption("自分が書いた、自分に効いたこと。忘れた頃に読み返して。")
        for _, row in _rec_df.sort_values("log_date", ascending=False).head(10).iterrows():
            dt = row["log_date"].strftime("%m-%d")
            st.markdown(f"- **{dt}** — {row['recovery']}")
        st.divider()

period = st.radio(
    "表示期間",
    ["直近7日", "直近30日", "直近90日", "全期間"],
    horizontal=True, index=1,
)
days_map = {"直近7日": 7, "直近30日": 30, "直近90日": 90, "全期間": 100000}
cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days_map[period])
view = df[df["log_date"] >= cutoff].copy()

if view.empty:
    st.info("この期間の記録はまだありません。")
    st.stop()

# --- ベースラインは「直近30日」で固定（表示期間を変えてもズレない） ---
BASELINE_WINDOW_DAYS = 30
BASELINE_MIN_SAMPLES = 7

_bl_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=BASELINE_WINDOW_DAYS)
_bl_df = df[df["log_date"] >= _bl_cutoff]

if len(_bl_df) >= BASELINE_MIN_SAMPLES:
    mean_mood = _bl_df["mood"].mean()
    std_mood = _bl_df["mood"].std(ddof=0) if len(_bl_df) > 1 else 0
    baseline_source = f"直近{BASELINE_WINDOW_DAYS}日（{len(_bl_df)}件）"
else:
    # 直近30日が少なすぎるときは全期間をベースラインにフォールバック
    mean_mood = df["mood"].mean()
    std_mood = df["mood"].std(ddof=0) if len(df) > 1 else 0
    baseline_source = f"全期間（{len(df)}件・データ蓄積中）"

lower, upper = mean_mood - 2 * std_mood, mean_mood + 2 * std_mood
view["anomaly"] = (view["mood"] < lower) | (view["mood"] > upper)
view["mood_ma7"] = view["mood"].rolling(7, min_periods=1).mean()

st.caption(
    f"📊 いつもの範囲は **{baseline_source}** の平均 ± 2σ から算出しています。"
    "表示期間を変えてもベースラインは固定です。"
)

fig = go.Figure()
fig.add_hrect(y0=lower, y1=upper,
              fillcolor="rgba(100,180,255,0.08)", line_width=0,
              annotation_text="いつもの範囲 (±2σ)",
              annotation_position="top left")
fig.add_hline(y=mean_mood, line_dash="dot", line_color="#888",
              annotation_text=f"平均 {mean_mood:.1f}",
              annotation_position="right")

normal = view[~view["anomaly"]]
fig.add_trace(go.Scatter(x=normal["log_date"], y=normal["mood"],
    mode="lines+markers", name="気分",
    line=dict(color="#4a90e2"), marker=dict(size=9)))

anom = view[view["anomaly"]]
if not anom.empty:
    fig.add_trace(go.Scatter(x=anom["log_date"], y=anom["mood"],
        mode="markers", name="いつもと違う日",
        marker=dict(color="#e74c3c", size=14,
                    symbol="circle-open", line=dict(width=3))))

fig.add_trace(go.Scatter(x=view["log_date"], y=view["mood_ma7"],
    mode="lines", name="7日移動平均",
    line=dict(color="#f39c12", width=2, dash="dash")))

fig.update_layout(
    yaxis=dict(range=[0.5, 10.5], title="気分"),
    xaxis=dict(title="日付"),
    hovermode="x unified",
    height=400, margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

if len(view) >= 3:
    st.subheader("📅 曜日別の気分")
    dow_df = dow_stats(view)
    fig_dow = go.Figure()
    fig_dow.add_trace(go.Bar(
        x=dow_df["曜日"].astype(str), y=dow_df["平均気分"],
        error_y=dict(type="data", array=dow_df["標準偏差"]),
        marker=dict(color=dow_df["平均気分"], colorscale="RdYlGn",
                    cmin=1, cmax=10),
        text=dow_df["記録数"].apply(lambda n: f"{n}件"),
        textposition="outside",
    ))
    fig_dow.update_layout(
        yaxis=dict(range=[0, 10.5], title="平均気分"),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    with st.expander("数値で見る"):
        st.dataframe(dow_df, use_container_width=True)
else:
    st.caption("曜日別の分析は記録3件以上で表示されます。")

if not anom.empty:
    with st.expander(f"🔴 いつもと違った日（{len(anom)}件）"):
        for _, row in anom.iterrows():
            dt = row["log_date"].strftime("%Y-%m-%d")
            w_emoji, w_label = describe_weather(row.get("weather_code"))
            extras = []
            if pd.notna(row.get("temperature")):
                extras.append(f"{row['temperature']:.1f}°C")
            if pd.notna(row.get("pressure")):
                extras.append(f"{row['pressure']:.0f}hPa")
            extras_str = " / ".join(extras)
            st.markdown(
                f"**{dt}** ｜ 気分 {row['mood']}／睡眠 {row['sleep_hours']}h "
                f"｜ {w_emoji}{w_label} {extras_str}"
            )
            if row["tags"]:
                _formatted_tags = " / ".join(
                    _format_tag(t) for t in _parse_tag_string(row["tags"])
                ) or row["tags"]
                st.caption(f"出来事: {_formatted_tags}")
            if row["note"]:
                st.write(row["note"])
            st.divider()

st.subheader("🔮 「翌日の気分」との相関")
st.caption("どの指標が翌日の気分と関係しそうか。絶対値が大きいほど影響が強い傾向。")

corr_df = correlations_with_next_mood(df)
if corr_df.empty:
    st.info("記録が5日分以上（連続して）溜まると分析できます。")
else:
    fig_corr = px.bar(
        corr_df.head(10),
        x="相関係数", y="特徴量", orientation="h",
        color="相関係数", color_continuous_scale="RdBu",
        range_color=[-1, 1],
        text=corr_df.head(10)["相関係数"].apply(lambda v: f"{v:+.2f}"),
    )
    fig_corr.update_layout(
        height=max(200, 40 * len(corr_df.head(10))),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    with st.expander("全項目の数値を見る"):
        st.dataframe(corr_df, use_container_width=True)

st.subheader("🤖 翌日の気分を予測してみる（実験）")
result = train_mood_predictor(df, min_samples=14)

if "importance" not in result:
    st.info(
        f"学習には連続した記録が{result['required']}日以上必要です。"
        f"現在使える日数：**{result['n_train']}日**"
    )
    st.caption("記録を続けていくと、ここに予測モデルと特徴量の寄与度が表示されます。")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("学習データ", f"{result['n_train']} 日")
    col2.metric("訓練R²", f"{result['train_r2']:.2f}")
    if result["cv_r2"] is not None:
        col3.metric("交差検証R²", f"{result['cv_r2']:.2f}",
                    help="-1〜1。1に近いほど予測が当たる")

    st.markdown("**特徴量の寄与度（翌日の気分を上げる/下げる要因）**")
    imp = result["importance"]
    fig_imp = px.bar(
        imp.head(10),
        x="効き方", y="特徴量", orientation="h",
        color="効き方", color_continuous_scale="RdBu",
        range_color=[-imp["効き方"].abs().max(), imp["効き方"].abs().max()],
        text=imp.head(10)["効き方"].apply(lambda v: f"{v:+.2f}"),
    )
    fig_imp.update_layout(
        height=max(200, 40 * len(imp.head(10))),
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "プラス：翌日の気分を上げる方向／マイナス：下げる方向。"
        "R²が低い場合は偶然の可能性あり。継続記録で精度が上がります。"
    )

    with st.expander("予測 vs 実測を見る"):
        pred_df = result["predictions"].copy()
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=pred_df["日付"], y=pred_df["実測"],
            mode="lines+markers", name="実測",
            line=dict(color="#4a90e2"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=pred_df["日付"], y=pred_df["予測"],
            mode="lines+markers", name="予測",
            line=dict(color="#e74c3c", dash="dash"),
        ))
        fig_pred.update_layout(
            yaxis=dict(range=[0.5, 10.5], title="気分（翌日）"),
            height=350, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

notes_df = view[view["note"].fillna("").str.len() > 0][
    ["log_date", "mood", "note"]
].sort_values("log_date", ascending=False)
if not notes_df.empty:
    st.subheader("最近のメモ")
    for _, row in notes_df.head(10).iterrows():
        dt = row["log_date"].strftime("%Y-%m-%d")
        st.markdown(f"**{dt}** (気分 {row['mood']})：{row['note']}")
