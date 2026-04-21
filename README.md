# 気分トラッカー（公開版）

スマホから外出先でも記録できる、マルチユーザー対応の気分トラッカー。
Streamlit Community Cloud × Supabase（PostgreSQL）で動作。

- **識別方式**：端末ごとのUUID（URLクエリ `?u=...` とローカル保存で自動復元）＋ 復元キー（32文字hex）
- **無料枠で運用可能**（Streamlit Cloud 無料・Supabase 無料プラン）

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

`.streamlit/secrets.toml` も `DATABASE_URL` 環境変数もなければ、
ローカルの `mood.db`（SQLite）にフォールバックして動作する。

## クラウド公開手順

### 1. Supabase プロジェクト作成

1. <https://supabase.com> にGitHubでログイン
2. 「New project」→ 任意の名前／パスワードを設定（**パスワードは必ず保管**）
3. Region は `Northeast Asia (Tokyo)` を推奨
4. プロジェクト作成後、**Settings → Database → Connection string → URI** をコピー
   - 「Connection pooling」の Session モード URI を推奨
   - 形式：`postgresql://postgres.xxxx:PASSWORD@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres`

### 2. GitHub リポジトリ作成

```bash
cd mood-tracker-public
git init
git add .
git commit -m "Initial commit"
# GitHub で `mood-tracker-public` リポジトリを作成し（owner: onepieceshintaro）：
git branch -M main
git remote add origin https://github.com/onepieceshintaro/mood-tracker-public.git
git push -u origin main
```

`.gitignore` で `.streamlit/secrets.toml` は除外されるので、DB接続文字列が漏れる心配はない。

### 3. Streamlit Community Cloud にデプロイ

1. <https://share.streamlit.io> にGitHubでログイン
2. 「New app」→ 先ほどのリポジトリ・ブランチ `main`・`app.py` を選択
3. 「Advanced settings」→ **Secrets** に以下を貼り付け：

   ```toml
   DATABASE_URL = "postgresql://postgres.xxxx:PASSWORD@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres"
   ```

4. Deploy をクリック → 数分で `https://<app-name>.streamlit.app` が発行される

### 4. 既存データの移行（任意・一回限り）

ローカルの `../mood-tracker/mood.db` に溜まったデータを Supabase に流し込む：

```bash
# .streamlit/secrets.toml を作成（secrets.toml.example を参考に）
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# DATABASE_URL を編集

# まずはドライラン
python migrate.py --dry-run

# 本番投入
python migrate.py
```

冪等なので何度流しても結果は同じ（user_id + log_date の複合キーで upsert）。

### 5. スマホでの使い方

1. Streamlit から発行されたURLをスマホで開く
2. 自動でUUIDが割り振られる（URL の `?u=...` 部分）
3. ブラウザで**このページをブックマーク**すれば、次回以降は開くだけで自動ログイン
4. 別端末でも使いたい場合は、サイドバーの「復元キー」を別端末で入力

## データベーススキーマ

```
mood_logs (
    user_id TEXT NOT NULL,
    log_date TEXT NOT NULL,
    mood INTEGER NOT NULL,
    sleep_hours DOUBLE PRECISION,
    energy INTEGER,
    note TEXT,
    tags TEXT,
    temperature DOUBLE PRECISION,
    weather_code INTEGER,
    precipitation DOUBLE PRECISION,
    pressure DOUBLE PRECISION,
    wake_time TEXT,
    PRIMARY KEY (user_id, log_date)
)
```

## ファイル構成

| ファイル | 役割 |
| --- | --- |
| `app.py` | Streamlit 本体（UI + 保存／読込） |
| `db.py` | SQLAlchemy エンジン。SQLite ↔ Postgres を吸収 |
| `_user.py` | ユーザーID・ニックネーム・サイドバーUI |
| `analysis.py` | 外れ値検出・相関分析（scikit-learn） |
| `weather.py` | Open-Meteo から天気・気圧を自動取得 |
| `migrate.py` | ローカルSQLite → Supabase への一回限り移行 |
| `.streamlit/secrets.toml.example` | 接続文字列のサンプル（本物は `.gitignore`） |

## トラブルシューティング

- **Supabase への接続が切れる**：`db.py` で `pool_pre_ping=True` を設定済み。自動再接続される。
- **`postgres://` で始まる URL が動かない**：`db.py` が自動で `postgresql+psycopg2://` に変換する。
- **無料枠が一時停止される**：Supabase の無料プロジェクトは 7 日間アクセスがないと pause される。復帰はダッシュボードのボタン一つ。
