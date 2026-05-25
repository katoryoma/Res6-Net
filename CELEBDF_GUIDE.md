# Celeb-DF デバイスセット対応ガイド

## 概要

このガイドでは、新しく統合された Celeb-DF データセット対応機能について説明します。

現在のプロジェクト構造を保持しながら、Celeb-DF データセットでのトレーニングとテストが可能になりました。

### 対応するモデルタイプ

- **3ch**: RGB画像のみ（ベースライン）
- **3ch_res**: Residual特徴のみ（2026-05-15 追加）
- **6ch**: RGB + Residual特徴の組み合わせ

## 主な機能

### 1. **動画ベースのデータ読み込み**
- Celeb-DF形式（`<videoID>/frame/` 構造）のデータセットに対応
- 各動画フレームを自動的に収集

### 2. **Cross-Validation (K-Fold)**
- 訓練データの20%を検証用に自動分割
- 複数のFoldで実験可能
- Fold番号でデータの分割が決まるため再現性を確保

### 3. **動画数の均衡化**
- Epochごとに合成動画数をReal動画数に合わせて自動サンプリング
- テスト時はすべての動画を使用

### 4. **既存プロジェクトとの互換性**
- 既存の `ImageDataset` クラスは変更なし
- 既存のトレーニングスクリプトは動作継続
- 新機能は新スクリプトで提供

## ファイル構成

```
プロジェクト/
├── utils/
│   └── config.py               # Celeb-DF設定を追加
├── train/
│   ├── train_celebdf.py        # Celeb-DF用トレーニングスクリプト
│   └── run_cross_validation_celebdf.py  # CV実行スクリプト
├── test/
│   ├── test_celebdf.py         # Celeb-DF用テストスクリプト
│   ├── test_celebdf_ensemble.py
│   └── test_celebdf_ensemble_cross_dataset.py
├── scripts/
│   └── inspect_celebdf_dataset.py  # データセット検証スクリプト
├── dataset/
│   └── dataset.py              # CelebDFDataset クラスを追加
└── CELEBDF_GUIDE.md            # このファイル
```

## セットアップ

### 1. データセット構造の確認

Celeb-DF データセットが以下の構造になっていることを確認してください：

```
!YOUR_CELEBDF_PATH!
├── real/
│   ├── train/
│   │   ├── <videoID_1>/
│   │   │   ├── frame_000000.jpg
│   │   │   ├── frame_000001.jpg
│   │   │   └── ...
│   │   ├── <videoID_2>/
│   │   └── ...
│   └── test/
│       ├── 00011/
│       │   ├── frame_000060.jpg
│       │   └── ...
│       └── ...
└── synthesis/
    ├── train/
    │   ├── id0_id1_0002/
    │   │   ├── frame_000060.jpg
    │   │   └── ...
    │   └── ...
    └── test/
        └── ...
```

### 2. データセット検証

データセット構造の整合性をチェックします：

```bash
python scripts/inspect_celebdf_dataset.py
```

出力例：
```
================================================================================
Inspecting Celeb-DF Train/Test Dataset
Root Directory: !YOUR_CELEBDF_PATH!\Celeb-DF\rsc\raw\real(synthesis)\train
================================================================================

Class           Videos     Total Frames   Avg Frames/Video   Min      Max
---------------------------------------------------------------------------
real            100        12500          125.0              50       200
synthesis       500        62500          125.0              50       200

Video Balance Check:
  Real videos:      100
  Synthesis videos: 500
  ✓ Synthesis has more videos (400 extra)
    → Will be sampled to match real videos during training
```

## 使用方法

### シングルFold トレーニング

**Fold 0 でトレーニング（6chモデル）：**

```bash
python train/train_celebdf.py 6ch 0
```

**Fold 1 でトレーニング（3chモデル）：**

```bash
python train/train_celebdf.py 3ch 1
```

**Fold 0 でトレーニング（3ch_res モデル - Residual のみ）：**

```bash
python train/train_celebdf.py 3ch_res 0
```

**コマンドラインオプション：**
```
python train/train_celebdf.py [model_type] [fold]

model_type: "3ch", "3ch_res" または "6ch" (デフォルト: 6ch)
fold:       Cross-Validation の折番号 (デフォルト: 0)
```

### シングルFold テスト

**Fold 0 でテスト（6chモデル）：**

```bash
python test/test_celebdf.py 6ch 0
```

**Fold 0 でテスト（3ch_res モデル）：**

```bash
python test/test_celebdf.py 3ch_res 0
```

出力例：
```
==================================================
Test Results
==================================================
Accuracy:  0.9234
Precision: 0.9165
Recall:    0.9156
F1 Score:  0.9160
==================================================
✓ Test results saved to: logs/celebdf_fold0/6ch/test_log.csv
```

### Cross-Validation 実行

**5-Fold Cross-Validation（6chモデル）：**

```bash
python train/run_cross_validation_celebdf.py 6ch 5
```

このコマンドで、Fold 0～4 の全フォルドに対して自動的にトレーニングとテストが実行されます。

**5-Fold Cross-Validation（3ch_res モデル - Residual のみ）：**

```bash
python train/run_cross_validation_celebdf.py 3ch_res 5
```

**コマンドラインオプション：**
```
python train/run_cross_validation_celebdf.py [model_type] [num_folds]

model_type: "3ch", "3ch_res" または "6ch" (デフォルト: 6ch)
num_folds:  Fold数 (デフォルト: 5)
```

### アンサンブルテスト

5つのFold（Fold 0～4）で学習したモデルを使用して、**単純平均によるアンサンブル推論**でテストを実行します。

**アンサンブルテスト実行（6chモデル）：**

```bash
python test/test_celebdf_ensemble.py 6ch
```

**アンサンブルテスト実行（3chモデル）：**

```bash
python test/test_celebdf_ensemble.py 3ch
```

**アンサンブルテスト実行（3ch_res モデル）：**

```bash
python test/test_celebdf_ensemble.py 3ch_res
```

**コマンドラインオプション：**
```
python test/test_celebdf_ensemble.py [model_type]

model_type: "3ch", "3ch_res" または "6ch" (デフォルト: 6ch)
```

**出力例：**
```
✓ Loaded model for fold 0
✓ Loaded model for fold 1
✓ Loaded model for fold 2
✓ Loaded model for fold 3
✓ Loaded model for fold 4
Ensemble Evaluation: 100%|████████████| 156/156 [00:45<00:00, 3.43it/s]

==================================================
Ensemble Test Results
==================================================
Accuracy:  0.9356
Precision: 0.9287
Recall:    0.9278
F1 Score:  0.9282
Macro Precision: 0.9287
Macro Recall:    0.9278
Macro F1:        0.9282
Weighted Precision: 0.9358
Weighted Recall:    0.9356
Weighted F1:        0.9356
ROC AUC:   0.9623
==================================================
Confusion Matrix:
[[1230   95]
 [  82 1156]]
==================================================
✓ Ensemble test results saved to: logs/celebdf_fold0/6ch/ensemble_test_log.csv
✓ Probabilities saved to: logs/celebdf_fold0/6ch/probabilities.csv
```

**処理の流れ：**

1. Fold 0～4 の学習済みモデルを読み込む
2. テストデータセットに対して各モデルで推論を実行
3. 各モデルの予測確率を平均化
4. 平均化された確率から最終予測クラスを決定
5. メトリクス（精度、適合率、再現率、F1スコア、ROC AUC等）を計算
6. 評価結果とサンプル毎の判別確率を保存

### クロスデータセット評価

複数の異なるテストデータセット（DF, DFD, F2F, FS, FSfter, NT）でアンサンブルモデルの性能を比較評価します。

**全データセットで評価（6chモデル）：**

```bash
python test/test_celebdf_ensemble_cross_dataset.py --model-type 6ch --dataset all
```

**特定のデータセットで評価（DFデータセット、3chモデル）：**

```bash
python test/test_celebdf_ensemble_cross_dataset.py --model-type 3ch --dataset DF
```

**特定のデータセットで評価（DFデータセット、3ch_resモデル）：**

```bash
python test/test_celebdf_ensemble_cross_dataset.py --model-type 3ch_res --dataset DF
```

**コマンドラインオプション：**
```
python test/test_celebdf_ensemble_cross_dataset.py [--model-type MODEL_TYPE] [--dataset DATASET]

--model-type: "3ch", "3ch_res" または "6ch" (デフォルト: 6ch)
--dataset:    "DF", "DFD", "F2F", "FS", "FSfter", "NT", または "all" (デフォルト: all)
```

**対応するテストデータセット：**

| データセット名 | パス | 説明 |
|---|---|---|
| DF | `!YOUR_FF_DATAPATH!/DF_test` | DeepFakes テストセット |
| DFD | `!YOUR_FF_DATAPATH!/dfd_test` | Deep Fake Detection テストセット |
| F2F | `!YOUR_FF_DATAPATH!/F2F_test` | Face2Face テストセット |
| FS | `!YOUR_FF_DATAPATH!/FS_test` | FaceSwap テストセット |
| FSfter | `!YOUR_FF_DATAPATH!/FSfter_test` | FaceSwap-After テストセット |
| NT | `!YOUR_FF_DATAPATH!/NT_test` | Neural Textures テストセット |

**出力例：**
```
================================================================================
Celeb-DF Ensemble Test - Cross Dataset Evaluation
================================================================================
Model Type: 6ch (RGB + Residual)
Device: cuda

Loading ensemble models...
✓ Loaded 5 models

================================================================================
Testing on DF dataset
================================================================================
Test dataset: 5200 samples
Test loader batches: 650

Evaluation: 100%|███████████████████████████████████████| 650/650 [02:15<00:00, 4.80it/s]

============================================================
DF Test Results
============================================================
Accuracy:             0.9156
Precision (macro):    0.9123
Recall (macro):       0.9145
F1-Score (macro):     0.9134
ROC AUC:              0.9654
...

================================================================================
Summary - Cross Dataset Evaluation Results
================================================================================
Dataset      Samples    Accuracy     Precision    Recall       F1-Score     ROC AUC    
--------------------------------------------------------------------------------
DF           5200       0.9156       0.9123       0.9145       0.9134       0.9654    
DFD          3400       0.8734       0.8612       0.8723       0.8667       0.9245    
F2F          2800       0.9234       0.9187       0.9201       0.9194       0.9732    
FS           4100       0.9045       0.8956       0.9023       0.8989       0.9534    
FSfter       3600       0.8856       0.8734       0.8812       0.8773       0.9423    
NT           2900       0.9312       0.9256       0.9289       0.9272       0.9812    

✓ Summary results saved to: logs/cross_dataset_evaluation_6ch.csv
✓ Probabilities saved to: logs/cross_dataset_probabilities/6ch/DF_probabilities.csv
✓ Probabilities saved to: logs/cross_dataset_probabilities/6ch/DFD_probabilities.csv
✓ Probabilities saved to: logs/cross_dataset_probabilities/6ch/F2F_probabilities.csv
...
```

## 出力ファイル

### チェックポイント

```
checkpoints/
├── celebdf_fold0/
│   ├── 3ch/
│   │   └── best_model.pth
│   ├── 3ch_res/
│   │   └── best_model.pth
│   └── 6ch/
│       └── best_model.pth
├── celebdf_fold1/
└── ...
```

### ログ・評価結果

```
logs/
├── celebdf_fold0/
│   ├── 3ch/
│   │   ├── train_log.csv          # トレーニング・検証ログ
│   │   └── test_log.csv           # テスト評価結果
│   ├── 3ch_res/
│   │   ├── train_log.csv
│   │   └── test_log.csv
│   └── 6ch/
│       ├── train_log.csv
│       └── test_log.csv
├── celebdf_fold1/
└── ...

# アンサンブルテスト結果
├── celebdf_ensemble/
│   ├── 3ch/
│   │   ├── test_log.csv           # アンサンブル評価結果
│   │   └── probabilities.csv      # 判別確率（各サンプル）
│   ├── 3ch_res/
│   │   ├── test_log.csv
│   │   └── probabilities.csv
│   └── 6ch/
│       ├── test_log.csv
│       └── probabilities.csv

# クロスデータセット評価結果
├── cross_dataset_evaluation_6ch.csv     # 全データセットの比較結果
└── cross_dataset_probabilities/
    ├── 3ch/
    │   ├── DF_probabilities.csv
    │   ├── DFD_probabilities.csv
    │   ├── F2F_probabilities.csv
    │   ├── FS_probabilities.csv
    │   ├── FSfter_probabilities.csv
    │   └── NT_probabilities.csv
    ├── 3ch_res/
    │   ├── DF_probabilities.csv
    │   └── ...
    └── 6ch/
        ├── DF_probabilities.csv
        └── ...
```

**train_log.csv:**
```
epoch,train_loss,train_acc,val_loss,val_acc
1,0.5234,0.7123,0.4892,0.7456
2,0.4123,0.7834,0.3956,0.8012
...
```

**test_log.csv:**
```
=== Basic Metrics ===
Metric,Value
Test Loss,0.2345
Test Accuracy,0.9234
ROC AUC,0.9512

=== Per-Class Accuracy ===
Class,Accuracy
real,0.9345
swap,0.9123
...
```

**probabilities.csv（判別確率）:**
```
Sample_ID,Real_Probability,Synthesis_Probability,Predicted_Class,True_Class,Correct
0,0.951234,0.048766,Real,Real,True
1,0.123456,0.876544,Synthesis,Synthesis,True
2,0.654321,0.345679,Real,Synthesis,False
...
```

**cross_dataset_evaluation_6ch.csv:**
```
Dataset,Samples,Accuracy,Precision,Recall,F1-Score,ROC AUC
DF,5200,0.9156,0.9123,0.9145,0.9134,0.9654
DFD,3400,0.8734,0.8612,0.8723,0.8667,0.9245
F2F,2800,0.9234,0.9187,0.9201,0.9194,0.9732
...
```

## CelebDFDataset クラスの仕様

### 初期化パラメータ

```python
dataset = CelebDFDataset(
    real_dir,              # Real クラスのディレクトリ
    synthesis_dir,         # Synthesis クラスのディレクトリ
    use_residual=True,     # Residualチャネルを使用するか
                           # True: RGB + Residual (6ch)
                           # "residual_only": Residual のみ (3ch_res)
                           # False: RGB のみ (3ch)
    is_train=True,         # 訓練モード/検証モード
    val_split=0.2,         # 検証用の割合
    fold=0,                # Fold番号（再現性のため）
    balance_videos=True    # 訓練時に動画数を均衡化するか
)

# 使用例
# RGB + Residual (6ch)
train_dataset = CelebDFDataset(
    real_dir="path/to/real/train",
    synthesis_dir="path/to/synthesis/train",
    use_residual=True,
    is_train=True,
    val_split=0.2,
    fold=0,
    balance_videos=True
)

# Residual のみ (3ch_res)
train_dataset_res_only = CelebDFDataset(
    real_dir="path/to/real/train",
    synthesis_dir="path/to/synthesis/train",
    use_residual="residual_only",
    is_train=True,
    val_split=0.2,
    fold=0,
    balance_videos=True
)
```

### 主なメソッド

```python
# エポック情報を設定（訓練時のみ）
dataset.set_epoch(epoch)

# データセットのサイズを取得
len(dataset)  # フレーム数

# サンプルを取得
x, label = dataset[idx]  # x: [C, H, W] tensor, label: int
```

### 主な特徴

1. **動的サンプリング**
   - 訓練時、Epochごとに異なる位置から画像を切り出す
   - 各Epochで異なるaugmentationが適用される

2. **ビデオ均衡化**
   - 訓練時にsynthesisビデオ数をrealに合わせてサンプリング
   - テスト時はすべてのビデオを使用

3. **Cross-Validation**
   - Fold番号とシードで決定的にデータを分割
   - 異なるFoldで異なるデータセット分割が得られる

## 設定のカスタマイズ

### config.py の関連設定

```python
# Celeb-DF ルートディレクトリ
CELEBDF_ROOT = r"!YOUR_CELEBDF_PATH!\Celeb-DF\rsc\raw"

# データセット構造（train/testディレクトリ）
CELEBDF_TRAIN_TEST_DIR = os.path.join(CELEBDF_ROOT, "real(synthesis)", "train")
CELEBDF_TEST_DIR = os.path.join(CELEBDF_ROOT, "real(synthesis)", "test")

# Cross-Validation設定
CELEBDF_VAL_SPLIT = 0.2        # 検証用割合（20%）
CELEBDF_BALANCE_VIDEOS = True  # 動画数均衡化の有効化
```

### パスのカスタマイズ例

別のデータセットパスを使用する場合：

```python
# train_celebdf.py の先頭で、configをインポート後に上書き
from config import *

CELEBDF_TRAIN_TEST_DIR = r"別のパス/train"
CELEBDF_TEST_DIR = r"別のパス/test"
```

## トラブルシューティング

### データセットが見つからない

```
❌ Root directory does not exist: !YOUR_DATAPATH!\...
```

**対策：**
- パスが正しいか確認
- config.py の `CELEBDF_ROOT` を確認

### フレームが見つからない

```
⚠️  No 'frame' directory in <videoID>
```

**対策：**
- ディレクトリ構造が `<videoID>/frame/` になっているか確認
- フレームファイルが `.png` または `.jpg` 形式か確認

### メモリ不足

**対策：**
- `config.py` の `BATCH_SIZE` を減らす
- `num_workers` を減らす（データローダーのパラメータ）

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # 8 から 4 に減らす
    shuffle=True,
    num_workers=2  # 4 から 2 に減らす
)
```

## 既存データセットとの並行使用

既存の3チャネル・3チャネルResidualsのみ・6チャネルデータセットと並行して使用できます：

```bash
# 既存データセットでの訓練
python train.py 6ch
python train.py 3ch
python train.py 3ch_res

# Celeb-DFでの訓練
python train_celebdf.py 6ch 0
python train_celebdf.py 3ch 0
python train_celebdf.py 3ch_res 0

# チェックポイントは別々に保存されます
# checkpoints/6ch/best_model.pth (既存)
# checkpoints/3ch/best_model.pth (既存)
# checkpoints/3ch_res/best_model.pth (既存)
# checkpoints/celebdf_fold0/6ch/best_model.pth (Celeb-DF)
# checkpoints/celebdf_fold0/3ch/best_model.pth (Celeb-DF)
# checkpoints/celebdf_fold0/3ch_res/best_model.pth (Celeb-DF)
```

## 実験例

### 1. 単一Fold検証

```bash
# Fold 0での6chモデル訓練
python train_celebdf.py 6ch 0

# テスト
python test_celebdf.py 6ch 0
```

### 2. 複数Fold検証

```bash
# 5-Fold Cross-Validation実行
python run_cross_validation_celebdf.py 6ch 5

# 結果は logs/celebdf_fold0, fold1, ... に保存
```

### 3. モデル比較

```bash
# 3ch モデルで5-Fold実行
python run_cross_validation_celebdf.py 3ch 5

# 3ch_res モデルで5-Fold実行
python run_cross_validation_celebdf.py 3ch_res 5

# 6ch モデルで5-Fold実行  
python run_cross_validation_celebdf.py 6ch 5

# 結果を比較
# logs/celebdf_fold{0-4}/3ch/test_log.csv
# logs/celebdf_fold{0-4}/3ch_res/test_log.csv
# logs/celebdf_fold{0-4}/6ch/test_log.csv
```

## まとめ

- **シンプル:** 既存プロジェクト構造を保持しながら新機能を追加
- **柔軟:** 複数Foldでの実験に対応
- **自動化:** Cross-Validation用のスクリプト完備
- **検証可能:** データセット検証スクリプトで事前チェック可能

詳しくは、各スクリプトのヘッダーコメントを参照してください。
