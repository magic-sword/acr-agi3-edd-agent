# 📋 Kaggle デプロイ & 検証コマンド リファレンス (Cheat Sheet)

本ドキュメントは、ローカル環境での検証・スコア計測から、Kaggle への自動デプロイ・ステータス確認までの運用コマンドをまとめたチートシートです。

---

## ⚡ 1. Kaggle CLI 自動デプロイ

### 🚀 ノートブックをビルドして Kaggle に一発デプロイ（Save & Run All）
```bash
docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --push
```
* **実行内容**:
  1. 最新の `src/` コードを自動で Base64 エンコードしてノートブック内に自己解凍コードとして内蔵
  2. メタデータ（GPU有効、インターネット無効）を生成
  3. `kaggle kernels push` で Kaggle サーバーへアップロードし、自動コミット実行を開始

### 🔍 Kaggle 上の実行ステータスを確認
```bash
docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --status
```
* ステータス遷移: `queued`（待機中） → `running`（実行中） → `complete`（完了）

### 📦 ソースコードを Kaggle Dataset としてアップロード（任意）
```bash
docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --push-dataset
```

---

## 🧪 2. ローカルでの事前検証 & スコア計測

### 📊 Kaggle オフライン環境シミュレーション & スコア計測
```bash
docker exec arc-agi3-dev python3 scripts/kaggle_local_simulation.py
```
* 迷路、オープン空間、溶岩帯トラップ、二重隘路、散乱障害物の 5 課題を走破し、クリア率・所要ステップ・実行時間を計測します。

### 📓 ノートブックのエンドツーエンド実行テスト
```bash
docker exec arc-agi3-dev jupyter nbconvert --to notebook --execute notebooks/submission_template.ipynb --output /tmp/test_executed.ipynb
```
* Jupyter ノートブックが上から下まで例外なく実行完走するかを事前にシミュレーションします。

### 🛡️ 単体・結合テスト実行 (EDD 防壁ゲート)
```bash
# 全テスト実行 (49件 PASS)
docker exec arc-agi3-dev pytest -q

# 提出パイプライン専用テスト
docker exec arc-agi3-dev pytest tests/test_submission_pipeline.py -v
```

### 🔍 静的コードチェック (Ruff リンター)
```bash
docker exec arc-agi3-dev uv run ruff check .
```

---

## 🔨 3. 自己完結型ノートブックの手動再ビルド

`src/` 配下のソースコードを変更した際、ノートブック内の埋め込みパッケージを手動で再更新する場合：
```bash
docker exec arc-agi3-dev python3 scripts/build_self_contained_notebook.py
```
* `notebooks/submission_template.ipynb` のセル 2 が最新の圧縮コードで自動更新されます。

---

## 🔑 4. Kaggle API 認証キー（401エラー時）の更新手順

Kaggle CLI から `401 Unauthorized` が返る場合、以下の 1 行でキーを反映できます：
```bash
# ブラウザで kaggle.json をダウンロード後、ホスト端末で実行:
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json && docker cp ~/.kaggle/kaggle.json arc-agi3-dev:/root/.kaggle/kaggle.json
```
反映後、再度デプロイコマンド（`--push`）を実行してください。
