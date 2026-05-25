import os
import sys
import torch

from torch.utils.data import DataLoader

from config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    SEED,
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
    get_checkpoint_dir,
    get_model_path,
    get_model_info,
    validate_model_type,
    DEFAULT_MODEL_TYPE,
    get_train_log_path
)
from dataset.dataset import ImageDataset
from models.resnet18 import build_model
from trainer import Trainer
from util import seed_everything


def main(model_type=None):
    """
    トレーニング実行
    
    Args:
        model_type (str, optional): モデルタイプ ("3ch" または "6ch")
                                   Noneの場合はデフォルト値を使用
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
    
    # =========================
    # Seed 固定
    # =========================
    seed_everything(SEED)
    
    # =========================
    # パス準備
    # =========================
    checkpoint_dir = get_checkpoint_dir(model_type)
    save_path = get_model_path(model_type)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"\nDevice: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Save path: {save_path}")
    
    # =========================
    # データセット読み込み
    # =========================
    print("\nLoading dataset...")
    train_dataset = ImageDataset(
        root_dir=TRAIN_DATA_PATH,
        use_residual=use_residual
    )
    val_dataset = ImageDataset(
        root_dir=VAL_DATA_PATH,
        use_residual=use_residual
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
    print(f"Model: {model.__class__.__name__}")
    print(f"Input channels: {in_channels}")
    
    # =========================
    # Trainer初期化
    # =========================
    print("\nInitializing trainer...")
    trainer = Trainer(train_loader, val_loader, model, DEVICE)
    
    # =========================
    # トレーニング実行
    # =========================
    print("\nStarting training (max {} epochs, early stopping patience=10)...".format(EPOCHS))
    train_log_path = get_train_log_path(model_type)
    trainer.fit(epochs=EPOCHS, save_path=save_path, log_path=train_log_path, patience=10)
    
    print(f"\nTraining completed!")
    print(f"Best model saved at: {save_path}")
    print(f"Training log saved at: {train_log_path}")


if __name__ == "__main__":
    # コマンドライン引数でモデルタイプを指定可能
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    
    main(model_type=model_type)
