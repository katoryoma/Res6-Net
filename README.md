# Res6-Net: ResNet18 for Deepfake Detection

ResNet18 をベースとしたディープフェイク検出モデルです。RGB画像、Residual特徴、およびそれらの組み合わせを使用した複数の入力形式をサポートしています。

## モデルタイプ

3つの異なる入力チャネル構成が利用可能です：

### 1. **3ch** (RGB のみ)
- **入力チャネル数**: 3 (RGB)
- **説明**: RGB画像のみを入力
- **用途**: ベースラインモデル
- **チェックポイント**: `checkpoints/3ch/`

```bash
python train.py 3ch
```

### 2. **3ch_res** (Residual のみ) ⭐ NEW
- **入力チャネル数**: 3 (Residual)
- **説明**: Residual特徴のみを入力
- **用途**: Residual信号の有効性を評価
- **チェックポイント**: `checkpoints/3ch_res/`

```bash
python train.py 3ch_res
```

### 3. **6ch** (RGB + Residual)
- **入力チャネル数**: 6 (RGB + Residual)
- **説明**: RGB画像と Residual特徴を結合した入力
- **用途**: 複合特徴を活用したモデル（デフォルト）
- **チェックポイント**: `checkpoints/6ch/`

```bash
python train.py 6ch
```

## Residual 特徴について

Residual 特徴は以下の手順で計算されます：

1. RGB画像をダウンサンプリング（384×384 → 128×128）
2. アップサンプリング（128×128 → 384×384）
3. 元の画像と復元画像の差分を計算
4. ログスケーリング増幅
5. 正規化

この手法は、画像の高周波成分（テクスチャなど）を強調し、ディープフェイクの工作痕跡を検出するのに有効です。

## トレーニング

### 基本的な使用方法

```bash
# デフォルト（6ch）でトレーニング
python train.py

# 3ch で実行
python train.py 3ch

# 3ch_res で実行
python train.py 3ch_res
```

## 設定

すべての設定は [config.py](config.py) で管理されています：

| 設定項目 | デフォルト値 |
|--------|-----------|
| バッチサイズ | 8 |
| エポック数 | 1000 |
| 学習率 | 1e-4 |
| 入力画像サイズ | 384×384 |
| Seed | 42 |

## ディレクトリ構成

```
.
├── models/
│   └── resnet18.py           # ResNet18 モデル定義
├── dataset/
│   ├── dataset.py            # データセット定義
│   └── transforms.py         # 画像変換処理
├── config.py                 # 設定ファイル
├── train.py                  # トレーニングスクリプト
├── trainer.py                # トレーナークラス
├── util.py                   # ユーティリティ関数
├── checkpoints/              # 保存されたモデル
│   ├── 3ch/
│   ├── 3ch_res/             # NEW
│   └── 6ch/
└── logs/                     # トレーニングログ
```

## ファイル説明

| ファイル | 説明 |
|--------|------|
| [config.py](config.py) | データセットパス、トレーニング設定、モデル設定を管理 |
| [train.py](train.py) | トレーニングエントリーポイント |
| [trainer.py](trainer.py) | トレーニングロジック（勾配更新、ログ記録など） |
| [models/resnet18.py](models/resnet18.py) | 入力チャネルに対応したResNet18実装 |
| [dataset/dataset.py](dataset/dataset.py) | `ImageDataset` と `CelebDFDataset` クラス |
| [dataset/transforms.py](dataset/transforms.py) | 画像変換・前処理 |
| [util.py](util.py) | ユーティリティ関数 |

## データセット

### 標準データセット

```
F:\2026_8025504_Kryoma\my_data\image\raw/
├── train/
│   ├── real/
│   └── swap/
├── validation/
│   ├── real/
│   └── swap/
└── test/
    ├── real/
    └── swap/
```

### Celeb-DF データセット

```
F:\2026_8025504_Kryoma\external_dataset\Celeb-DF\rsc\raw/
├── real/
│   ├── train/
│   └── test/
└── synthesis/
    ├── train/
    └── test/
```

## 最近の変更

### 2026-05-15
- **新機能**: Residual のみの3チャネルモデル (`3ch_res`) を追加
  - RGB画像を使用せず、Residual特徴だけで学習
  - `config.py` の `MODEL_CONFIG` に新しい設定を追加
  - `dataset/dataset.py` に対応ロジックを実装
  - 詳細は [config.py](config.py) の `MODEL_CONFIG` 参照

### モデル入力チャネルの対応

| モデルタイプ | conv1.weight シェイプ | データセット出力 |
|---|---|---|
| 3ch | (64, 3, 7, 7) | (3, 384, 384) |
| 3ch_res | (64, 3, 7, 7) | (3, 384, 384) |
| 6ch | (64, 6, 7, 7) | (6, 384, 384) |

## ライセンス・使用データセット

このプロジェクトは以下のデータセットを使用する想定です。使用する際は、各データセットの利用規約を遵守してください。

### Celeb-DF Dataset
- **出典**: [Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics](https://github.com/yuezunli/celeb-deepfakeforensics)
- **利用条件**: 研究目的のみ（商用利用は不可）
- **論文**: Li et al. (2020). "The Eyes Tell All: Detecting Political Orientation from Eye Movement Data" [実際の論文情報は公式リポジトリを参照]

### Face Forensics++ (FF++)
- **出典**: [FaceForensics++: Learning to Detect Manipulated Facial Images](https://github.com/ondyari/FaceForensics)
- **利用条件**: 学術研究・非営利目的のみ
- **論文**: Rössler et al. (2019)

### ⚠️ 重要事項
- このコードを商用目的で使用する場合は、各データセットの利用規約を確認してください
- Celeb-DF と Face Forensics++ は研究目的のための条件付きアクセスとなっています
- データセット所有者の明示的な許可を得ずに商用利用することはできません



