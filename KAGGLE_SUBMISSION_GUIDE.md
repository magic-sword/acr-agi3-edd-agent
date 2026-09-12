# 🚀 Kaggle リーダーボード提出ガイド (ACR-AGI-3)

本ガイドは、`acr-agi3-edd-agent` の推論パイプラインを Kaggle リーダーボードへ投稿し、実スコアを計測するための標準手順書です。

---

## ⚡ 方法 1: コマンドラインからの 1 発デプロイ (推奨)

Kaggle CLI 自動デプロイスクリプト（`scripts/deploy_kaggle.py`）を使って、ノートブックのビルドから Kaggle へのプッシュ（Save & Run All 実行開始）までをコマンド一発で実行できます。

### 1. ノートブックのプッシュ（デプロイ）
```bash
docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --push
```
* **動作内容**:
  1. `src/` 配下の最新コードを Base64 自動自己解凍コードとしてノートブック（`submission_template.ipynb`）内に内蔵ビルド
  2. `deploy/kaggle_kernel/` にメタデータを自動生成
  3. `kaggle kernels push` で Kaggle サーバーへ自動アップロードし、コミット実行（Save & Run All）を開始

### 2. 実行ステータスの確認
```bash
docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --status
```
* Kaggle 上での実行状態（`queued` → `running` → `complete`）を確認できます。

### 3. リーダーボード提出
Kaggle 上で実行完了後、Kaggle ノートブックの出力ページから **「Submit to Competition」** をクリックして完了です。

---

## 🖥️ 方法 2: Kaggle Web UI からの提出手順

### ステップ 1: Kaggle Notebook のインポート
1. ARC-AGI-3 コンペティションページを開き、**「Code」タブ →「New Notebook」** をクリック。
2. メニューの **「File」→「Import Notebook」** から、ローカルの `notebooks/submission_template.ipynb` をアップロード。
   *(※最新のノートブックは内部に自己解凍コードを含んでいるため、追加のデータセットアップロードは不要です！)*

---

### ステップ 2: ノートブックの環境設定（右側パネル）
* **Accelerator**: `GPU T4 x2` または `GPU P100`（GPU を有効化）
* **Internet**: **`Off`**（※提出要件: 完全オフライン必須）

---

### ステップ 3: 実行と提出（Save & Submit）
1. 右上の **「Save Version」** をクリック。
2. **Version Type**: **`Save & Run All (Commit)`** を選択して **「Save」** を実行。
3. 正常完了後、**「Submit to Competition」** ボタンを押して提出！

---

## 🔑 Kaggle API 認証エラー（401）が出る場合の対処法
Kaggle CLI から `401 - Unauthorized` が返る場合は、以下の手順で API トークンを更新してください：
1. [Kaggle Account Settings](https://www.kaggle.com/settings) を開く
2. 「API」セクションの **「Create New Token」** をクリック（`kaggle.json` がダウンロードされます）
3. ダウンロードしたファイルをプロジェクト内に配置：
   ```bash
   cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   docker cp ~/.kaggle/kaggle.json arc-agi3-dev:/root/.kaggle/kaggle.json
   ```
4. 再度 `docker exec arc-agi3-dev python3 scripts/deploy_kaggle.py --push` を実行してください。
