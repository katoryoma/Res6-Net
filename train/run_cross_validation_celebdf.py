"""
Celeb-DF データセット用の Cross-Validation runner

すべてのfoldに対してトレーニングとテストを実行します。
"""

import os
import sys

from .train_celebdf import main as train_main
from test.test_celebdf import main as test_main


def run_cross_validation(model_type="6ch", num_folds=5):
    """
    Cross-Validation を実行
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
        num_folds (int): Fold数
    """
    print(f"\n{'='*60}")
    print(f"Cross-Validation for Celeb-DF Dataset")
    print(f"Model Type: {model_type}")
    print(f"Number of Folds: {num_folds}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for fold in range(num_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{num_folds}")
        print(f"{'='*60}")
        
        # トレーニング
        print(f"\n--- Training Fold {fold} ---")
        try:
            train_main(model_type, fold)
        except Exception as e:
            print(f"❌ Training failed for fold {fold}: {e}")
            continue
        
        # テスト
        print(f"\n--- Testing Fold {fold} ---")
        try:
            test_main(model_type, fold)
        except Exception as e:
            print(f"❌ Testing failed for fold {fold}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else "6ch"
    num_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    run_cross_validation(model_type, num_folds)
