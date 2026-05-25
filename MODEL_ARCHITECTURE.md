# Res6-Net: モデルアーキテクチャ詳細ガイド

ResNet18をベースとしたディープフェイク検出モデルの詳細なアーキテクチャドキュメントです。

---

## 📋 目次

1. [概要](#概要)
2. [モデルアーキテクチャ](#モデルアーキテクチャ)
3. [入力チャネル形式](#入力チャネル形式)
4. [入力前処理パイプライン](#入力前処理パイプライン)
5. [モデルタイプの比較](#モデルタイプの比較)
6. [トレーニングパイプライン](#トレーニングパイプライン)
7. [プロジェクト構成](#プロジェクト構成)

---

## 概要

**Res6-Net** は、以下の特徴を備えたディープフェイク検出システムです：

- **ベースモデル**: ResNet18（ImageNetで事前学習済み）
- **出力**: バイナリ分類（Real / Fake）
- **入力形式**: 複数の入力チャネル構成に対応
- **主要機能**: Residual特徴を活用したマルチチャネル入力

---

## モデルアーキテクチャ

### 基本構造

```
┌─────────────────────────────────────────────┐
│          入力層 (Variable Channels)         │
│  ┌──────────────────────────────────────┐   │
│  │ 3ch: RGB only                        │   │
│  │ 3ch_res: Residual only              │   │
│  │ 6ch: RGB + Residual (concatenated)  │   │
│  └──────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│  Conv1: 7×7, Stride=2, Padding=3            │
│  入力チャネル数に応じて動的に設定            │
│  出力: [64, H/2, W/2]                       │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         MaxPool2d: 3×3, Stride=2            │
│         出力: [64, H/4, W/4]                 │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│    ResNet18 BasicBlock Stack                │
│    ├─ Layer1 (2×BasicBlock): [64, H/4]      │
│    ├─ Layer2 (2×BasicBlock): [128, H/8]     │
│    ├─ Layer3 (2×BasicBlock): [256, H/16]    │
│    └─ Layer4 (2×BasicBlock): [512, H/32]    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│    GlobalAveragePooling                     │
│    出力: [512]                               │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│    全結合層 (Fully Connected)                 │
│    入力: 512 → 出力: 2 (Real/Fake)          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│    出力: [batch_size, 2]                     │
│    Softmax (推論時)                          │
└──────────────────────────────────────────────┘
```

### 重要な実装詳細

#### 1. 動的Conv1層

標準的なResNet18では入力チャネル数が3に固定されていますが、このプロジェクトでは可変です：

```python
# 元のconv1層の重みを取得
old_conv = model.conv1

# 新しいconv1層を作成（入力チャネル数を変更）
model.conv1 = nn.Conv2d(
    in_channels,      # 3, 3, または 6
    64,              # 常に64出力チャネル
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

# 事前学習済み重みの転移学習
with torch.no_grad():
    if in_channels >= 3:
        model.conv1.weight[:, :3] = old_conv.weight
    if in_channels >= 6:
        model.conv1.weight[:, 3:6] = old_conv.weight
```

**処理**:
- RGB チャネルには元の ImageNet 事前学習重みを使用
- Residual チャネルには RGB 重みをコピーして初期化
- これにより、事前学習の利点を活かしながら、新しい入力形式に対応

#### 2. 全結合層の適応

```python
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
```

デフォルトでは1000クラスの出力を持つImageNetモデルを、2クラス分類に適応させます。

---

## 入力チャネル形式

### 1. **3ch** (RGB のみ)

| パラメータ | 値 |
|---------|-----|
| 入力チャネル数 | 3 |
| 説明 | RGB画像のみ |
| 用途 | ベースラインモデル（標準的なCNN） |
| チェックポイント | `checkpoints/celebdf_fold*/3ch/` |

**入力形状**: `[Batch, 3, 384, 384]`

```
RGB画像 (384×384×3)
    ↓
Conv1（3→64チャネル）
    ↓
ResNet18処理
    ↓
出力: [Batch, 2]
```

### 2. **3ch_res** (Residual のみ) ⭐

| パラメータ | 値 |
|---------|-----|
| 入力チャネル数 | 3 |
| 説明 | Residual特徴のみ |
| 用途 | Residual信号の独立した効果を評価 |
| チェックポイント | `checkpoints/celebdf_fold*/3ch_res/` |

**入力形状**: `[Batch, 3, 384, 384]`（Residual特徴）

```
RGB画像 (384×384×3)
    ↓
Residual特徴抽出（詳細は以下参照）
    ↓
Residual画像 (384×384×3)
    ↓
Conv1（3→64チャネル）
    ↓
ResNet18処理
    ↓
出力: [Batch, 2]
```

### 3. **6ch** (RGB + Residual) 🌟

| パラメータ | 値 |
|---------|-----|
| 入力チャネル数 | 6 |
| 説明 | RGB画像とResidual特徴の結合入力 |
| 用途 | **推奨** - 両方の情報を活用したモデル |
| チェックポイント | `checkpoints/celebdf_fold*/6ch/` |

**入力形状**: `[Batch, 6, 384, 384]`

```
RGB画像 (384×384×3) + Residual画像 (384×384×3)
    ↓
Concatenate → [384×384×6]
    ↓
Conv1（6→64チャネル）
    ↓
ResNet18処理
    ↓
出力: [Batch, 2]
```

---

## 入力前処理パイプライン

### Residual特徴の計算アルゴリズム

Residual特徴は、以下のステップで RGB 画像から抽出されます：

#### ステップ1: ダウンサンプリング（Lossy圧縮のシミュレーション）

```
元の画像: 384×384×3
    ↓ (F.interpolate - Bilinear)
圧縮画像: 128×128×3
```

**目的**: 高周波成分（テクスチャ詳細）を削除し、低周波成分のみを保持

**実装**:
```python
down = F.interpolate(
    x.unsqueeze(0),
    size=(128, 128),
    mode="bilinear",
    align_corners=False
)
```

#### ステップ2: アップサンプリング（復元）

```
圧縮画像: 128×128×3
    ↓ (F.interpolate - Bilinear)
復元画像: 384×384×3
```

**目的**: 圧縮画像を元のサイズに復元

**実装**:
```python
up = F.interpolate(
    down,
    size=(size, size),
    mode="bilinear",
    align_corners=False
)
```

#### ステップ3: 残差計算

```
Residual = 元の画像 - 復元画像
```

**目的**: 圧縮により失われた高周波成分（ディープフェイク工作痕跡）を抽出

**実装**:
```python
residual = x - up
```

**図解**:
```
元画像
  │
  ├─→ [ダウンサウンプリング] → 128×128 → [アップサンプリング] → 復元画像
  │                                                                    │
  │←───────────────────────────[差分計算]─────────────────────────────┤
  
  Residual = 元画像 - 復元画像
```

#### ステップ4: ログスケーリング増幅

```python
# Residualの絶対値にログスケーリングを適用
residual = torch.log(torch.abs(residual) + 1e-6)
```

**目的**:
- 小さな値の詳細を強調
- 値の分布を圧縮し、異なるスケールの特徴を均等に扱う
- 学習の安定性向上

#### ステップ5: 正規化

```python
# 各チャネルを [-1, 1] の範囲に正規化
residual = residual / (residual.max() + 1e-6) * 2 - 1
```

**目的**:
- 入力値を標準的な範囲に統一
- モデルの学習を安定化

### パイプライン全体図

```
入力画像 (RGB, 384×384)
    │
    ├─→ Data Augmentation (Geometric)
    │   ├─ Random Crop
    │   ├─ Random Flip
    │   └─ Random Rotation
    │
    ├─→ ToTensor() → [0, 1] 范围
    │
    ├─→ [3ch の場合] そのまま使用
    │
    └─→ [3ch_res, 6ch の場合]
        │
        ├─→ Residual特徴抽出
        │   ├─ ダウンサンプリング (384→128)
        │   ├─ アップサンプリング (128→384)
        │   ├─ 差分計算
        │   ├─ ログスケーリング
        │   └─ 正規化
        │
        ├─→ [3ch_res] Residual のみ を出力
        │
        └─→ [6ch] RGB + Residual を Concatenate して出力
```

---

## モデルタイプの比較

### パラメータ数

| モデルタイプ | Conv1入力 | Conv1出力 | パラメータ数 |
|-----------|----------|---------|-----------|
| 3ch | 3 | 64 | ~11.2M |
| 3ch_res | 3 | 64 | ~11.2M |
| 6ch | 6 | 64 | ~11.4M |

**注**: Conv1層のパラメータ変化
- 3ch: 3×64×49 = 9,408
- 6ch: 6×64×49 = 18,816

### 計算複雑度（FLOP）

```
入力サイズ: 384×384

Conv1層:
  - 3ch:  2 × 384² × 49 / 4 ≈ 7.1M FLOPs
  - 6ch:  2 × 384² × 98 / 4 ≈ 14.3M FLOPs

ResNet18本体: ≈ 1.8B FLOPs（入力チャネルに依存しない）

合計:
  - 3ch:  ≈ 1.807B FLOPs
  - 6ch:  ≈ 1.814B FLOPs
```

---

## トレーニングパイプライン

### 学習フロー

```
1. データセット読み込み
   └─ Real / Fake ビデオフレーム分類

2. バッチサンプリング
   ├─ Batch Size: 8
   └─ 毎エポック異なるフレーム位置をサンプリング

3. 前処理
   ├─ Geometric Augmentation (Random Crop, Flip, Rotation)
   ├─ ToTensor
   └─ Residual特徴抽出（6chの場合）

4. モデル順伝播
   ├─ Conv1 → 64チャネル
   ├─ ResNet18 4層
   └─ FC層 → [Batch, 2]

5. 損失計算
   └─ CrossEntropyLoss

6. 逆伝播
   └─ Adam Optimizer (lr=1e-4)

7. 検証
   ├─ 検証データセットでの精度評価
   └─ Early Stopping判定
```

### 学習パラメータ

| パラメータ | 値 |
|----------|-----|
| 最適化手法 | Adam |
| 学習率 | 1×10⁻⁴ |
| 損失関数 | CrossEntropyLoss |
| バッチサイズ | 8 |
| 最大エポック数 | 1000 |
| Early Stopping Patience | 10 |
| 画像サイズ | 384×384 |

### Early Stopping機構

```python
if val_acc > best_acc:
    best_acc = val_acc
    patience_counter = 0
    save_model(model, save_path)  # ベストモデル保存
else:
    patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break  # トレーニング終了
```

---

## プロジェクト構成

### ディレクトリ構造

```
Res6-Net/
├── models/
│   └── resnet18.py              # モデル定義
├── dataset/
│   ├── dataset.py               # データセットクラス
│   └── transforms.py            # 前処理・増幅
├── trainer.py                   # トレーニングループ
├── config.py                    # 設定ファイル
├── train.py / train_celebdf.py  # トレーニングスクリプト
├── test.py / test_celebdf.py    # 推論スクリプト
├── util.py                      # ユーティリティ関数
├── checkpoints/                 # チェックポイント保存先
│   └── celebdf_fold{0-4}/
│       ├── 3ch/
│       ├── 3ch_res/
│       └── 6ch/
└── logs/                        # ログファイル
    └── celebdf_fold{0-4}/
        ├── 3ch/
        ├── 3ch_res/
        └── 6ch/
```

### 主要ファイルの役割

#### `models/resnet18.py`
- **機能**: ResNet18モデルの構築と初期化
- **関数**: `build_model(in_channels, num_classes)`
- **入力**: 入力チャネル数（3または6）
- **出力**: 学習可能なモデルインスタンス

#### `dataset/dataset.py`
- **機能**: 画像データセット読み込みとResidual特徴抽出
- **クラス**: `ImageDataset`, `SixChannelDataset`
- **出力**: (画像, ラベル) ペア

#### `dataset/transforms.py`
- **機能**: 幾何学的データ増幅（Geometric Augmentation）
- **含む**: Random Crop, Flip, Rotation

#### `trainer.py`
- **機能**: トレーニング・検証ループの実装
- **クラス**: `Trainer`
- **メソッド**: `train_one_epoch()`, `validate()`, `fit()`

#### `config.py`
- **機能**: 全体設定の管理
- **設定内容**: 
  - データセットパス
  - モデルパラメータ
  - 学習ハイパーパラメータ
  - モデルタイプ定義

---

## 推論時の処理

### 予測フロー

```
テスト画像
    │
    ├─→ 前処理（学習時と同じ）
    │   ├─ ToTensor
    │   └─ Residual特徴抽出（必要に応じて）
    │
    ├─→ モデル順伝播
    │   └─ 出力: logits [2]
    │
    ├─→ Softmax確率変換
    │   └─ [P(Real), P(Fake)]
    │
    └─→ 予測
        └─ argmax → {0: Real, 1: Fake}
```

### 出力フォーマット

```python
# ロジット出力
logits = model(image)  # shape: [2]

# 確率化
probs = torch.softmax(logits, dim=0)
# probs[0]: Real確率
# probs[1]: Fake確率

# 予測クラス
pred = torch.argmax(logits)  # 0=Real, 1=Fake
```

---

## 性能指標

### 評価メトリクス

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision_Fake = TP / (TP + FP)

Recall_Fake = TP / (TP + FN)

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### 交差検証スキーム

```
Celeb-DF データセット
    │
    ├─ Fold0: Train(Real1-600, Syn1-600) | Val(Real601-700, Syn601-700)
    ├─ Fold1: Train(Real1-500, Syn1-500) | Val(Real501-600, Syn501-600)
    ├─ Fold2: Train(Real1-500, Syn1-500) | Val(Real501-600, Syn501-600)
    ├─ Fold3: Train(Real1-500, Syn1-500) | Val(Real501-600, Syn501-600)
    └─ Fold4: Train(Real1-500, Syn1-500) | Val(Real501-600, Syn501-600)
```

5分割交差検証により、モデルの汎化性能を評価します。

---

## トラブルシューティング

### よくある問題

**Q: 6chモデルが3chより精度が低い場合**
- A: Residual特徴の抽出方法を確認してください
- Downsampling → Upsampling の間隔を調整
- ログスケーリングの係数を変更

**Q: メモリ不足エラー**
- A: バッチサイズを削減（8 → 4）
- 画像サイズを縮小（384 → 224）

**Q: Early Stopping が早すぎる**
- A: Patienceパラメータを増加（10 → 20）
- 学習率を調整（1e-4 → 5e-5）

---

## 参考文献

- ResNet: He, K., et al. "Deep Residual Learning for Image Recognition" (2015)
- Deepfake Detection: Li, Y., et al. "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics" (2020)

---

**更新日**: 2026年5月22日  
**バージョン**: 1.0

