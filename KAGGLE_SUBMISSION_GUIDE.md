# 🚀 Kaggle リーダーボード提出ガイド (ACR-AGI-3)

本ガイドは、`acr-agi3-edd-agent` の推論パイプラインを Kaggle リーダーボードへ投稿し、実スコアを計測するための標準手順書です。

---

## 📦 提出用アセットの準備状況

以下の提出用アセットが生成済みです：

| アセット | パス | 用途 |
| :--- | :--- | :--- |
| **提出用ノートブック** | `notebooks/submission_template.ipynb` | Kaggle Notebook にインポートして提出するメインファイル |
| **ソースコード配布アーカイブ** | `dist/acr_agi3_source.tar.gz` | Kaggle Dataset としてアップロード可能なソース一式 (108KB) |
| **スタンドアロン提出スクリプト** | `dist/run_submission.py` | 単体スクリプト提出または実行ランナー |

---

## 🛠️ Kaggle リーダーボード提出手順（推奨: Notebook 提出）

### ステップ 1: ソースコードを Kaggle Dataset としてアップロード
Kaggle Notebook がオフライン環境で `acr_agi3` パッケージを読み込めるようにします。

1. [Kaggle Datasets](https://www.kaggle.com/datasets) を開き、**「+ New Dataset」** をクリック。
2. ローカルの `dist/acr_agi3_source.tar.gz` をドラッグ＆ドロップでアップロード。
3. タイトルを `acr-agi3-source` として作成（Create）。

---

### ステップ 2: Kaggle Notebook の作成・インポート
1. ARC-AGI-3 コンペティションページを開き、**「Code」タブ →「New Notebook」** をクリック。
2. メニューの **「File」→「Import Notebook」** から、ローカルの `notebooks/submission_template.ipynb` をアップロード。

---

### ステップ 3: ノートブックの環境設定（右側パネル）
以下の設定を確認してください：

* **Accelerator**: `GPU T4 x2` または `GPU P100`（GPU を有効化）
* **Internet**: **`Off`**（※提出要件: オフライン必須）
* **Inputs (Data)**:
  - コンペティション公式データセット（自動マウント）
  - ステップ 1 で作成した `acr-agi3-source`
  - *(必要に応じて)* ローカル LLM モデル重み（`Qwen2.5-Coder-1.5B-Instruct` 等の Kaggle Dataset）

---

### ステップ 4: コードの実行と提出（Save & Submit）
1. ノートブックの全セルが正常に実行され、最下部のセルで：
   ```
   === Submission Verification ===
   Total Tasks in Submission: ...
   🎉 Submission ready for Kaggle Leaderboard!
   ```
   と表示され、`/kaggle/working/submission.json` が生成されていることを確認。
2. 右上の **「Save Version」** をクリック。
3. **Version Type**: **`Save & Run All (Commit)`** を選択して **「Save」** を実行。
4. 実行ログが正常完了（Status: Complete）したら、左メニューの **「Output」** または右側パネルの **「Submit to Competition」** ボタンを押してリーダーボードへ提出！

---

## 🛡️ フェイルセーフ & 安全機構
* **ゲシュタルト直感ナビゲーター内蔵**:
  - LLM が未接続またはタイムアウトした場合でも、最短経路探索（BFS/ゲシュタルト直感）が自動起動し、スコア 0 を回避して確実にステージクリアします。
* **Jupyter イベントループ安全化**:
  - `nest_asyncio` による保護が入っており、セル実行時の非同期例外を遮断します。
