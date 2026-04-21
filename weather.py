"""Open-Meteo APIから気象データを取得する。APIキー不要・無料。"""
from datetime import date
from collections import Counter
from statistics import mean

import requests

# 兵庫県明石市（デフォルト位置）
DEFAULT_LAT = 34.6417
DEFAULT_LON = 134.9969

WEATHER_LABELS = {
    0:  ("☀️", "快晴"),
    1:  ("🌤️", "晴れ"),
    2:  ("⛅", "薄曇り"),
    3:  ("☁️", "曇り"),
    45: ("🌫️", "霧"),
    48: ("🌫️", "霧"),
    51: ("🌦️", "霧雨"),
    53: ("🌦️", "霧雨"),
    55: ("🌦️", "強い霧雨"),
    61: ("🌧️", "小雨"),
    63: ("🌧️", "雨"),
    65: ("🌧️", "強い雨"),
    71: ("🌨️", "小雪"),
    73: ("🌨️", "雪"),
    75: ("🌨️", "大雪"),
    77: ("🌨️", "霧雪"),
    80: ("🌧️", "にわか雨"),
    81: ("🌧️", "にわか雨"),
    82: ("🌧️", "強いにわか雨"),
    85: ("🌨️", "にわか雪"),
    86: ("🌨️", "強いにわか雪"),
    95: ("⛈️", "雷雨"),
    96: ("⛈️", "雷雨（雹）"),
    99: ("⛈️", "雷雨（雹）"),
}


def describe_weather(code):
    if code is None or (isinstance(code, float) and code != code):  # NaN
        return ("❓", "不明")
    return WEATHER_LABELS.get(int(code), ("❓", "不明"))


_JP_SUFFIXES = ("市", "区", "町", "村", "都", "府", "県")


def _geocode_once(name: str, timeout: int):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "ja",
                    "format": "json"},
            timeout=timeout,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return None
        top = results[0]
        return {
            "name": top.get("name"),
            "admin1": top.get("admin1"),
            "country": top.get("country"),
            "latitude": top["latitude"],
            "longitude": top["longitude"],
        }
    except Exception:
        return None


def geocode_city(name: str, timeout: int = 10):
    """市名・地名から緯度経度を解決する。Open-Meteo Geocoding API（キー不要）。
    見つかれば dict {name, admin1, country, latitude, longitude} を返す。
    失敗時は None。

    検索順（日本の地名で同名多数の対策）：
      - 接尾辞付き（明石市/新宿区/東京都 など）: そのまま → 外して再試行
      - 接尾辞なし（明石/神戸/大阪 など）: 「市」を足して → そのまま
    """
    if not name or not name.strip():
        return None
    q = name.strip()

    # 検索バリエーションを組み立て（重複除去しつつ順序維持）
    candidates = []
    matched_suffix = next((s for s in _JP_SUFFIXES if q.endswith(s)), None)
    if matched_suffix and len(q) > len(matched_suffix):
        stripped = q[: -len(matched_suffix)]
        # 例: 「大阪府」→ ['大阪府', '大阪', '大阪市']
        # 例: 「京都府」→ ['京都府', '京都', '京都市']
        candidates = [q, stripped, stripped + "市"]
    else:
        # 例: 「神戸」→ ['神戸市', '神戸']
        # 例: 「京都」（都が末尾でも念のため +市 も試す）→ ['京都市', '京都']
        candidates = [q + "市", q]

    # 最後の砦：q のまま + "市"（「京都」→「京都市」対策。接尾辞が「都」で
    # 誤判定されるケースを救う）
    if not q.endswith("市"):
        candidates.append(q + "市")

    seen = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        hit = _geocode_once(cand, timeout)
        if hit:
            return hit
    return None


def fetch_weather(log_date: date, lat: float = DEFAULT_LAT,
                  lon: float = DEFAULT_LON, timeout: int = 10):
    """指定日の気象データを取得。
    - 過去 → archive-api
    - 今日以降 → forecast-api
    失敗時は None を返す。"""
    today = date.today()
    is_past = log_date < today
    base_url = (
        "https://archive-api.open-meteo.com/v1/archive" if is_past
        else "https://api.open-meteo.com/v1/forecast"
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(log_date),
        "end_date": str(log_date),
        "hourly": "temperature_2m,weather_code,precipitation,pressure_msl",
        "timezone": "Asia/Tokyo",
    }

    try:
        r = requests.get(base_url, params=params, timeout=timeout)
        r.raise_for_status()
        h = r.json().get("hourly", {})

        temps = [t for t in h.get("temperature_2m", []) if t is not None]
        codes = [int(c) for c in h.get("weather_code", []) if c is not None]
        precips = [p for p in h.get("precipitation", []) if p is not None]
        pressures = [p for p in h.get("pressure_msl", []) if p is not None]

        if not temps:
            return None

        return {
            "temperature": round(mean(temps), 1),
            "weather_code": Counter(codes).most_common(1)[0][0] if codes else None,
            "precipitation": round(sum(precips), 1),
            "pressure": round(mean(pressures), 1) if pressures else None,
        }
    except Exception:
        return None
