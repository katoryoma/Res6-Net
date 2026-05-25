"""
Celeb-DF データセット用トレーニングスクリプト

使用方法:
    python train_celebdf.py [model_type] [fold]
    
    model_type: "3ch" または "6ch" (デフォルト: 6ch)
    fold: Cross-Validation の折番号 (デフォルト: 0)
    
例:
    python train_celebdf.py 6ch 0  # 6ch モデルでfold 0を実行
    python train_celebdf.py 3ch 1  # 3ch モデルでfold 1を実行
"""

import os
import sys
import torch

from torch.utils.data import DataLoader

from utils.config import (
    CELEBDF_ROOT,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
    get_celebdf_checkpoint_dir,
    get_celebdf_model_path,
    get_model_info,
    validate_model_type,
    DEFAULT_MODEL_TYPE,
    get_celebdf_train_log_path
)
from dataset.dataset import CelebDFDataset
from models.resnet18 import build_model
from utils.trainer import Trainer
from utils.util import seed_everything


def main(model_type=None, fold=0):
    """
    Celeb-DFデータセットでのトレーニング実行
    
    Args:
        model_type (str, optional): モデルタイプ ("3ch" または "6ch")
                                   Noneの場合はデフォルト値を使用
        fold (int, optional): Cross-Validation の折番号
    """
    # =========================
    # モデルタイプの設定
    # =========================
    if model_type is None:
        model_type = DEFAULT_MODEL_TYPE
    
    try:
        validate_model_type(model_type)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    model_info = get_model_info(model_type)
    in_channels = model_info["in_channels"]
    use_residual = model_info["use_residual"]
    
    print(f"Model Type: {model_type} ({model_info['description']})")
    print(f"Cross-Validation Fold: {fold}")
    
    # =========================
    # Seed 固定
    # =========================
    seed_everything(SEED)
    
    # =========================
    # パス準備
    # =========================
    checkpoint_dir = get_celebdf_checkpoint_dir(model_type, f"celebdf_fold{fold}")
    save_path = get_celebdf_model_path(model_type, f"celebdf_fold{fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Save path: {save_path}")
    
    # =========================
    # データセット読み込み
    # =========================
    print("\nLoading Celeb-DF dataset...")
    train_dataset = CelebDFDataset(
        real_dir=os.path.join(CELEBDF_ROOT, "real", "train"),
        synthesis_dir=os.path.join(CELEBDF_ROOT, "synthesis", "train"),
        use_residual=use_residual,
        is_train=True,
        fold=fold,
        num_folds=5,  # 5-fold cross-validation
        balance_videos=True
    )
    
    val_dataset = CelebDFDataset(
        real_dir=os.path.join(CELEBDF_ROOT, "real", "train"),
        synthesis_dir=os.path.join(CELEBDF_ROOT, "synthesis", "train"),
        use_residual=use_residual,
        is_train=False,
        fold=fold,
        num_folds=5,
        balance_videos=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # =========================
    # モデル構築
    # =========================
    print("\nBuilding model...")
    model = build_model(in_channels=in_channels, num_classes=2)
    print(f"Input channels: {in_channels}")
    
    # =========================
    # トレーナー準備
    # =========================
    print("\nPreparing trainer...")
    trainer = Trainer(train_loader, val_loader, model, DEVICE)
    
    # =========================
    # トレーニング実行
    # =========================
    print(f"\nStarting training (max {EPOCHS} epochs, early stopping patience=10)...")
    log_path = get_celebdf_train_log_path(model_type, f"celebdf_fold{fold}")
    trainer.fit(epochs=EPOCHS, save_path=save_path, log_path=log_path, patience=10)
    
    print(f"\n✓ Training completed!")
    print(f"Best model saved to: {save_path}")
    print(f"Logs saved to: {log_path}")


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    main(model_type, fold)
