# =========================
# Configuration File
# =========================

import os
import torch

# =========================
# データセット設定
# =========================
DATASET_ROOT = r"!YOUR_DATAPATH!\my_data\image\raw"

TRAIN_DATA_PATH = os.path.join(DATASET_ROOT, "train")
VAL_DATA_PATH = os.path.join(DATASET_ROOT, "validation")
TEST_DATA_PATH = os.path.join(DATASET_ROOT, "test")

# =========================
# Celeb-DF データセット設定
# =========================
# ディレクトリ構造:
# CELEBDF_ROOT/
# ├── real/
# │   ├── train/
# │   └── test/
# └── synthesis/
#     ├── train/
#     └── test/
CELEBDF_ROOT = r"!YOUR_DATAPATH!\external_dataset\Celeb-DF\rsc\raw"

# 複数テストデータセットの定義
CROSS_DATASET_TEST_DATASETS = {
    "DF": r"!YOUR_DATAPATH!\external_dataset\DF_test\rsc\raw",
    "DFD": r"!YOUR_DATAPATH!\external_dataset\dfd_test\rsc\raw",
    "F2F": r"!YOUR_DATAPATH!\external_dataset\F2F_test\rsc\raw",
    "FS": r"!YOUR_DATAPATH!\external_dataset\FS_test\rsc\raw",
    "FSfter": r"!YOUR_DATAPATH!\external_dataset\FSfter_test\rsc\raw",
    "NT": r"!YOUR_DATAPATH!\external_dataset\NT_test\rsc\raw",
}

# Cross-Validation設定
CELEBDF_VAL_SPLIT = 0.2  # 訓練データの20%を検証用に
CELEBDF_BALANCE_VIDEOS = True  # Epochごとにsynthesis動画数をrealに合わせるか

# =========================
# トレーニング設定
# =========================
SEED = 42
BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 1e-4

# =========================
# 画像設定
# =========================
IMAGE_SIZE = 384  # 入力画像サイズ (pixels)

# =========================
# モデル設定
# =========================
# 3ch: RGB のみ（ベースライン）
# 6ch: RGB + Residual
DEFAULT_MODEL_TYPE = "6ch"
NUM_CLASSES = 2

# モデルタイプに応じた入力チャネル数
MODEL_CONFIG = {
    "3ch": {
        "in_channels": 3,
        "use_residual": False,
        "description": "RGB only (Baseline)"
    },
    "3ch_res": {
        "in_channels": 3,
        "use_residual": "residual_only",
        "description": "Residual only"
    },
    "6ch": {
        "in_channels": 6,
        "use_residual": True,
        "description": "RGB + Residual"
    }
}

# =========================
# デバイス設定
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# チェックポイント設定
# =========================
CHECKPOINT_ROOT = "checkpoints"

# モデルタイプ別のチェックポイントディレクトリ
def get_checkpoint_dir(model_type="6ch"):
    """
    モデルタイプに応じたチェックポイントディレクトリを取得
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
    
    Returns:
        str: チェックポイントディレクトリパス
    """
    return os.path.join(CHECKPOINT_ROOT, model_type)


def get_model_path(model_type="6ch", filename="best_model.pth"):
    """
    モデルファイルのパスを取得
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
        filename (str): ファイル名
    
    Returns:
        str: モデルファイルのパス
    """
    return os.path.join(get_checkpoint_dir(model_type), filename)


# =========================
# ログ設定
# =========================
LOG_DIR = "logs"

def get_log_dir(model_type="6ch"):
    """
    ログディレクトリを取得
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
    
    Returns:
        str: ログディレクトリパス
    """
    return os.path.join(LOG_DIR, model_type)


def get_train_log_path(model_type="6ch"):
    """
    訓練ログCSVファイルのパスを取得
    
    Args:
        model_type (str): モデルタイプ
    
    Returns:
        str: 訓練ログCSVファイルのパス
    """
    return os.path.join(get_log_dir(model_type), "train_log.csv")


def get_test_log_path(model_type="6ch"):
    """
    テストログCSVファイルのパスを取得
    
    Args:
        model_type (str): モデルタイプ
    
    Returns:
        str: テストログCSVファイルのパス
    """
    return os.path.join(get_log_dir(model_type), "test_log.csv")


# =========================
# Celeb-DF用の関数
# =========================
def get_celebdf_checkpoint_dir(model_type="6ch", dataset_name="celebdf"):
    """
    Celeb-DFデータセット用のチェックポイントディレクトリを取得
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
        dataset_name (str): データセット名
    
    Returns:
        str: チェックポイントディレクトリパス
    """
    return os.path.join(CHECKPOINT_ROOT, dataset_name, model_type)


def get_celebdf_model_path(model_type="6ch", dataset_name="celebdf", filename="best_model.pth"):
    """
    Celeb-DFデータセット用のモデルファイルパスを取得
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
        dataset_name (str): データセット名
        filename (str): ファイル名
    
    Returns:
        str: モデルファイルのパス
    """
    return os.path.join(get_celebdf_checkpoint_dir(model_type, dataset_name), filename)


def get_celebdf_log_dir(model_type="6ch", dataset_name="celebdf"):
    """
    Celeb-DFデータセット用のログディレクトリを取得
    
    Args:
        model_type (str): モデルタイプ
        dataset_name (str): データセット名
    
    Returns:
        str: ログディレクトリパス
    """
    return os.path.join(LOG_DIR, dataset_name, model_type)


def get_celebdf_train_log_path(model_type="6ch", dataset_name="celebdf"):
    """
    Celeb-DFデータセット用の訓練ログCSVファイルのパスを取得
    
    Args:
        model_type (str): モデルタイプ
        dataset_name (str): データセット名
    
    Returns:
        str: 訓練ログCSVファイルのパス
    """
    return os.path.join(get_celebdf_log_dir(model_type, dataset_name), "train_log.csv")


def get_celebdf_test_log_path(model_type="6ch", dataset_name="celebdf"):
    """
    Celeb-DFデータセット用のテストログCSVファイルのパスを取得
    
    Args:
        model_type (str): モデルタイプ
        dataset_name (str): データセット名
    
    Returns:
        str: テストログCSVファイルのパス
    """
    return os.path.join(get_celebdf_log_dir(model_type, dataset_name), "test_log.csv")


# =========================
# 検証用の関数
# =========================
def validate_model_type(model_type):
    """
    モデルタイプが有効かチェック
    
    Args:
        model_type (str): モデルタイプ
    
    Raises:
        ValueError: 無効なモデルタイプの場合
    """
    if model_type not in MODEL_CONFIG:
        raise ValueError(
            f"Invalid model_type: {model_type}. "
            f"Must be one of {list(MODEL_CONFIG.keys())}"
        )


def get_model_info(model_type="6ch"):
    """
    モデルの設定情報を取得
    
    Args:
        model_type (str): モデルタイプ
    
    Returns:
        dict: モデルの設定情報
    """
    validate_model_type(model_type)
    return MODEL_CONFIG[model_type]


# =========================
# デバッグ用: 設定値の確認
# =========================
if __name__ == "__main__":
    print("=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    
    print("\n[Dataset Paths]")
    print(f"Train: {TRAIN_DATA_PATH}")
    print(f"Val:   {VAL_DATA_PATH}")
    print(f"Test:  {TEST_DATA_PATH}")
    
    print("\n[Training Config]")
    print(f"Seed:         {SEED}")
    print(f"Batch Size:   {BATCH_SIZE}")
    print(f"Epochs:       {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    print("\n[Device]")
    print(f"Device: {DEVICE}")
    
    print("\n[Model Types]")
    for model_type, config in MODEL_CONFIG.items():
        print(f"  {model_type}: {config['description']}")
        print(f"    - Input Channels: {config['in_channels']}")
        print(f"    - Checkpoint: {get_model_path(model_type)}")
    
    print("\n" + "=" * 50)
