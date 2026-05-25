import os
import sys
import torch

from torch.utils.data import DataLoader

from utils.config import (
    TEST_DATA_PATH,
    SEED,
    BATCH_SIZE,
    DEVICE,
    get_model_path,
    get_model_info,
    validate_model_type,
    DEFAULT_MODEL_TYPE,
    get_test_log_path
)
from dataset.dataset import ImageDataset
from models.resnet18 import build_model
from utils.util import set_seed, evaluate, save_test_log


def main(model_type=None):
    """
    テスト実行
    
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
    set_seed(SEED)
    
    # =========================
    # パス準備
    # =========================
    model_path = get_model_path(model_type)
    
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Model path: {model_path}")
    
    # =========================
    # テストデータセット読み込み
    # =========================
    print("\nLoading test dataset...")
    test_dataset = ImageDataset(
        root_dir=TEST_DATA_PATH,
        use_residual=use_residual
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
        print(f"Please train the model first by running: python train.py {model_type}")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    print("✓ Model weights loaded")
    
    # =========================
    # テスト実行
    # =========================
    print("\nRunning evaluation on test set...")
    results = evaluate(
        model, 
        test_loader, 
        DEVICE, 
        num_classes=2
    )
    
    # =========================
    # 結果表示
    # =========================
    print("\n" + "=" * 70)
    print(f"TEST RESULTS ({model_type})")
    print("=" * 70)
    
    print(f"\nBasic Metrics:")
    print(f"  Test Loss:     {results['loss']:.4f}")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    for class_name, acc in results['per_class_acc'].items():
        print(f"  {class_name}: {acc:.4f}")
    
    class_names = {0: "real", 1: "swap"}
    print(f"\nPrecision, Recall, F1 (Per-Class):")
    for i in range(2):
        class_name = class_names.get(i, f"class_{i}")
        print(f"  {class_name}:")
        print(f"    Precision: {results['precision'][i]:.4f}")
        print(f"    Recall:    {results['recall'][i]:.4f}")
        print(f"    F1-Score:  {results['f1'][i]:.4f}")
    
    print(f"\nMacro Averages:")
    print(f"  Precision: {results['precision_macro']:.4f}")
    print(f"  Recall:    {results['recall_macro']:.4f}")
    print(f"  F1-Score:  {results['f1_macro']:.4f}")
    
    print(f"\nWeighted Averages:")
    print(f"  Precision: {results['precision_weighted']:.4f}")
    print(f"  Recall:    {results['recall_weighted']:.4f}")
    print(f"  F1-Score:  {results['f1_weighted']:.4f}")
    
    print(f"\nROC AUC Score: {results['roc_auc']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("=" * 70)
    
    # =========================
    # ログ保存
    # =========================
    test_log_path = get_test_log_path(model_type)
    save_test_log(results, test_log_path)
    print(f"\nTest log saved at: {test_log_path}")


if __name__ == "__main__":
    # コマンドライン引数でモデルタイプを指定可能
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    
    main(model_type=model_type)
