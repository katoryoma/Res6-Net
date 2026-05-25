"""
Celeb-DF データセット用テストスクリプト

使用方法:
    python test_celebdf.py [model_type] [fold]
    
    model_type: "3ch" または "6ch" (デフォルト: 6ch)
    fold: Cross-Validation の折番号 (デフォルト: 0)
    
例:
    python test_celebdf.py 6ch 0
    python test_celebdf.py 3ch 1
"""

import os
import sys
import torch

from torch.utils.data import DataLoader

from config import (
    CELEBDF_ROOT,
    SEED,
    BATCH_SIZE,
    DEVICE,
    get_celebdf_model_path,
    get_model_info,
    validate_model_type,
    DEFAULT_MODEL_TYPE,
    get_celebdf_test_log_path
)
from dataset.dataset import CelebDFDataset
from models.resnet18 import build_model
from util import set_seed, evaluate, save_test_log


def main(model_type=None, fold=0):
    """
    Celeb-DFデータセットでのテスト実行
    
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
    set_seed(SEED)
    
    # =========================
    # パス準備
    # =========================
    model_path = get_celebdf_model_path(model_type, f"celebdf_fold{fold}")
    
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Model path: {model_path}")
    
    # =========================
    # テストデータセット読み込み
    # =========================
    print("\nLoading Celeb-DF test dataset...")
    
    # テスト時は全ビデオを使用（検証用の分割を使わない）
    test_dataset = CelebDFDataset(
        real_dir=os.path.join(CELEBDF_ROOT, "real", "test"),
        synthesis_dir=os.path.join(CELEBDF_ROOT, "synthesis", "test"),
        use_residual=use_residual,
        is_train=False,  # テストモード
        fold=fold,
        num_folds=5,
        balance_videos=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Test loader batches: {len(test_loader)}")
    
    # =========================
    # モデル構築
    # =========================
    print("\nBuilding model...")
    model = build_model(in_channels=in_channels, num_classes=2)
    print(f"Input channels: {in_channels}")
    
    # =========================
    # モデル重み読み込み
    # =========================
    print(f"\nLoading model weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"Please train the model first by running: python train_celebdf.py {model_type} {fold}")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("✓ Model weights loaded")
    
    # =========================
    # テスト実行
    # =========================
    print("\nEvaluating on test dataset...")
    metrics = evaluate(model, test_loader, DEVICE)
    
    print(f"\n{'='*50}")
    print("Test Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"{'='*50}")
    
    # =========================
    # ログ保存
    # =========================
    log_path = get_celebdf_test_log_path(model_type, f"celebdf_fold{fold}")
    save_test_log(metrics, log_path)
    print(f"\n✓ Test results saved to: {log_path}")


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    main(model_type, fold)
