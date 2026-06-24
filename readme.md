# Image Classification System

## システム概要

ResNet-18（ImageNet事前学習済みモデル）をベースにした画像分類システムです．
クラス名の設定から訓練画像のアップロード・モデル学習・予測・可視化まで，
コードを書かずにブラウザ上のGUIで操作できます．

バックエンドは FastAPI（Python），フロントエンドはバニラ JS の SPA です．
サーバー起動後，`http://localhost:8000` でアクセスできます．

主な特徴：
- 全層ファインチューニングと最終層のみの線形プローブを選択可能
- データ拡張・差分学習率・CosineAnnealing スケジューラによる精度向上
- 複数ファイルをドラッグ＆ドロップで一括アップロード・一括予測
- 予測確率はクラス設定順で表示・結果はページ遷移後も保持
- CPU推論をINT8量子化で高速化
- 学習の進捗（Epoch・Loss）をリアルタイム表示

## 機能一覧

| ページ | 機能 |
|--------|------|
| Upload Training Images | クラス名の設定・訓練画像アップロード・画像の削除 |
| Train Model | ResNet-18のファインチューニング（全層 または 最終層のみ） |
| Predict | 複数画像の一括分類・確率表示・結果キャッシュ |
| Saliency Map Analysis | 勾配ベース特徴重要度マップの可視化 |
| Save Model | 学習済みモデル（クラス情報付き）のダウンロード |
| Load Model | 保存済みモデルの読み込みと再利用 |

---

## 起動方法

### Windows

**前提条件:** Python 3.9以上（Anaconda可）または https://python.org からインストール

```powershell
cd Project\PetClassifier

# 初回（仮想環境作成＋パッケージインストール）
.\start.ps1

# 2回目以降（高速起動）
.\start.ps1 -NoInstall
```

**実行ポリシーエラーが出た場合:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### Mac

**前提条件:** Python 3.9以上

```bash
# Python が未インストールの場合（Homebrew）
brew install python
```

```bash
cd Project/PetClassifier

# 実行権限を付与（初回のみ）
chmod +x start.sh

# 初回（仮想環境作成＋パッケージインストール）
./start.sh

# 2回目以降（高速起動）
./start.sh --no-install
```

> **Apple Silicon (M1/M2/M3) の場合:** PyTorch が MPS バックエンド（GPU）を自動認識し，CPU より高速に学習します．

---

### Linux

**前提条件:** Python 3.9以上

```bash
# Python が未インストールの場合
sudo apt install python3 python3-venv   # Ubuntu/Debian
# sudo dnf install python3              # Fedora/RHEL
```

```bash
cd Project/PetClassifier

# 実行権限を付与（初回のみ）
chmod +x start.sh

# 初回（仮想環境作成＋パッケージインストール）
./start.sh

# 2回目以降（高速起動）
./start.sh --no-install
```

---

### 起動の流れ

起動スクリプトを実行すると，ブラウザが自動的に「起動中...」のローディング画面を開きます．
サーバーの準備が整うと自動的に `http://localhost:8000` へリダイレクトされます．

初回起動時は PyTorch や ResNet-18 の事前学習済み重みのダウンロードで数十秒かかる場合があります．

---

### 仮想環境について

起動スクリプトはプロジェクトフォルダ内に `.venv/` フォルダを作成します．
これはパッケージをPC全体から隔離するための**フォルダ**です．

不要になったら削除できます：

```powershell
# Windows
Remove-Item -Recurse -Force .venv
```
```bash
# Mac / Linux
rm -rf .venv
```

削除後に `start.ps1` / `start.sh` を再実行すれば再作成されます．

---

## 使用方法

### Step 1: クラス名を設定する

「Upload Training Images」ページを開きます．

- デフォルトで `class_1`，`class_2` の2クラスが設定されています
- テキストボックスに任意のクラス名を入力します（例: `dog`，`cat`）
- 「**＋ クラスを追加**」ボタンで3クラス以上にも対応できます
- 「**クラス名を確定する**」をクリックします

> クラス名を変更すると学習済みモデルはリセットされます（再学習が必要です）．

### Step 2: 訓練画像をアップロードする

各クラスに **1枚以上** の画像（PNG / JPG / JPEG）をアップロードします．
ドラッグ＆ドロップまたはクリックで選択できます．クラスごとに異なる枚数でも構いません．
Train Model ページでは現在の枚数が表示されます．

アップロード済みの画像を削除したい場合：
- クラスごとの「**🗑️ 削除**」ボタンでそのクラスの画像をすべて削除
- ページ下部の「**🗑️ 全クラスの画像をまとめて削除**」で一括削除

> 削除後は学習済みモデルが自動的にリセットされます（再学習が必要です）．

### Step 3: モデルを学習する

「Train Model」ページで学習方式を選んでボタンをクリックします．

| ボタン | 方式 | 特徴 |
|--------|------|------|
| **全層トレーニング** | 全層ファインチューニング | 精度が出やすい・学習時間が長い |
| **最終層のみトレーニング** | 線形プローブ（FC層のみ更新） | 高速・過学習しにくい・少枚数向き |

学習中は「モデルを構築中 → 学習中（Epoch X / Y，Loss: Z）→ 量子化中」の進捗がリアルタイム表示されます．

学習には以下の精度向上技術が自動適用されます：
- **データ拡張**: ランダムクロップ・左右反転・色変動・回転を毎エポック適用し，過学習を抑制
- **差分学習率**: 全層学習時は Backbone を `lr=1e-4`，FC 層を `lr=1e-3` で別々に最適化
- **CosineAnnealing**: 学習率をエポック終盤に向けてなめらかに減衰
- **クラス不均衡補正**: 枚数が少ないクラスの損失を自動で大きく重みづけ

> **初回実行時のみ**，ResNet-18の事前学習済み重みをインターネットからダウンロードします（約44MB）．
> 2回目以降はキャッシュが使用されます．

### Step 4: 予測する

「Predict」ページで分類したい画像を複数枚まとめてドロップまたは選択します．
アップロードすると自動的に推論が実行され，各クラスの確率が棒グラフで表示されます．

- 確率はクラス設定時の順序で表示されます（確率の高い順ではありません）
- 一度予測した結果は「Saliency Map Analysis」ページに移動して戻っても保持されます
- 新しい画像をアップロードすると結果が更新されます

### Step 5: Saliency Map 分析を確認する

「Saliency Map Analysis」ページで，予測に影響した画像領域をヒートマップで確認できます．

- Predict ページの各予測結果にある「🗺 Saliency Map」ボタンを押すと Saliency Map ページへ自動遷移します
- 計算完了後，ヒートマップが自動表示されます
- **赤・黄・白の領域**: 予測に影響を与えた部分
- **黒い領域**: 影響が少ない部分

### Step 6: モデルを保存・再利用する

「Save Model」ページの「⬇ Download Model」ボタンで学習済みモデルを `.pth` ファイルとしてダウンロードできます．
クラス名情報もファイルに含まれており，「Load Model」ページでドロップして読み込むとクラス名が自動的に復元されます．

---

## 技術仕様

| 項目 | 内容 |
|------|------|
| バックエンド | FastAPI + uvicorn |
| フロントエンド | バニラ JS SPA（ダークテーマ） |
| ベースモデル | ResNet-18（ImageNet事前学習済み） |
| 学習方法 | 全層ファインチューニング **または** 線形プローブ（最終FC層のみ）を選択可 |
| 最適化手法 | Adam（全層: Backbone `lr=1e-4` / FC `lr=1e-3`，最終層のみ: `lr=0.001`），10エポック |
| LR スケジューラ | CosineAnnealingLR（終盤に向けてなめらかに減衰） |
| データ拡張 | RandomResizedCrop・RandomHorizontalFlip・ColorJitter・RandomRotation（学習時のみ） |
| クラス不均衡補正 | クラスごとの枚数比に応じた損失の重みづけ（CrossEntropyLoss weight） |
| 入力サイズ | 224×224px（自動リサイズ） |
| 正規化 | ImageNet標準（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]） |
| 推論高速化 | 学習後にINT8動的量子化モデルを生成（CPU推論専用） |
| 特徴量可視化 | Gradient-based Saliency Map（`autograd.grad` で入力勾配のみ計算） |
| モデル保存形式 | PyTorch checkpoint（state_dict + クラス名リスト） |
| GPU対応 | CUDA（Windows/Linux）・MPS（Apple Silicon）・CPU の順で自動選択 |

### 分類フロー

```
入力画像
  → Resize(224×224)
  → ToTensor + Normalize
  → ResNet-18 Backbone（特徴抽出）
  → 全結合層 fc（クラス数出力）
  → Softmax → 各クラスの確率
```

### Saliency Map フロー

```
入力テンソル（requires_grad=True）
  → モデル forward pass
  → autograd.grad で入力テンソルの勾配のみを計算
    （モデルパラメータの勾配は計算しないため高速）
  → 勾配絶対値を取得
  → RGBチャネル方向で最大値集約 → Saliency Map（2D）
  → パーセンタイル正規化 → 元画像にオーバーレイして表示
```

---

## ファイル構成

```
PetClassifier/
├── api.py              # FastAPI バックエンド・全 API エンドポイント
├── LICENSE             # ライセンスファイル
├── model.py            # ResNet-18定義・学習ループ・量子化
├── utils.py            # 画像前処理・ファイル保存・ディレクトリ管理
├── requirements.txt    # 依存パッケージ一覧（Win/Mac/Linux共通）
├── start.ps1           # Windows 起動スクリプト（PowerShell）
├── start.sh            # Mac/Linux 起動スクリプト（bash）
├── readme.md            # このドキュメント
├── static/
│   ├── index.html      # フロントエンド SPA（バニラ JS）
│   └── loading.html    # 起動待機画面（サーバー応答後に自動リダイレクト）
└── uploads/            # アップロード画像の保存先（自動生成）
    ├── class_1/
    └── class_2/
```
