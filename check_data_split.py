"""
Train/Val データ分割の比率確認スクリプト
"""
import os
from config import CELEBDF_ROOT
from dataset.dataset import CelebDFDataset

def check_split(model_type, fold):
    """データ分割の比率を確認"""
    print(f"\n{'='*70}")
    print(f"Model Type: {model_type}, Fold: {fold}")
    print(f"{'='*70}")
    
    from config import get_model_info
    model_info = get_model_info(model_type)
    use_residual = model_info["use_residual"]
    
    # Train Dataset
    train_dataset = CelebDFDataset(
        real_dir=os.path.join(CELEBDF_ROOT, "real", "train"),
        synthesis_dir=os.path.join(CELEBDF_ROOT, "synthesis", "train"),
        use_residual=use_residual,
        is_train=True,
        fold=fold,
        num_folds=5,
        balance_videos=True
    )
    
    # Val Dataset
    val_dataset = CelebDFDataset(
        real_dir=os.path.join(CELEBDF_ROOT, "real", "train"),
        synthesis_dir=os.path.join(CELEBDF_ROOT, "synthesis", "train"),
        use_residual=use_residual,
        is_train=False,
        fold=fold,
        num_folds=5,
        balance_videos=True
    )
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    total = train_size + val_size
    
    train_ratio = train_size / total * 100 if total > 0 else 0
    val_ratio = val_size / total * 100 if total > 0 else 0
    
    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}")
    print(f"Total samples: {total}")
    print(f"Train ratio:   {train_ratio:.1f}%")
    print(f"Val ratio:     {val_ratio:.1f}%")
    
    # 理想値との比較
    ideal_train_ratio = 80
    ideal_val_ratio = 20
    
    print(f"\nIdeal Train: {ideal_train_ratio}%, Actual: {train_ratio:.1f}%")
    print(f"Ideal Val:   {ideal_val_ratio}%, Actual: {val_ratio:.1f}%")
    
    # ビデオ数の確認
    print(f"\nTrain videos - Real: {len(train_dataset.train_videos['real'])}, "
          f"Synthesis: {len(train_dataset.train_videos['synthesis'])}")
    print(f"Val videos - Real: {len(val_dataset.val_videos['real'])}, "
          f"Synthesis: {len(val_dataset.val_videos['synthesis'])}")

if __name__ == "__main__":
    print("Checking data split for fold 0")
    
    check_split("3ch", 0)
    check_split("6ch", 0)
