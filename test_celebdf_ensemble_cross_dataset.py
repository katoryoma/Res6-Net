"""
Celeb-DF アンサンブルテスト - クロスデータセット評価

6種類の異なるテストデータセット（DF, DFD, F2F, FS, FSfter, NT）で
アンサンブルモデルの性能を評価するスクリプト

使用方法:
    python test_celebdf_ensemble_cross_dataset.py [model_type] [dataset_name]
    
    model_type: "3ch" または "6ch" (デフォルト: 6ch)
    dataset_name: "DF", "DFD", "F2F", "FS", "FSfter", "NT", または "all" (デフォルト: all)
    
例:
    python test_celebdf_ensemble_cross_dataset.py 6ch all      # 全データセットで評価
    python test_celebdf_ensemble_cross_dataset.py 6ch DF       # DF データセットのみ
    python test_celebdf_ensemble_cross_dataset.py 3ch F2F      # F2F データセット、3ch モデル
"""

import os
import sys
import numpy as np
import torch
import pandas as pd

from torch.utils.data import DataLoader
from collections import defaultdict

from config import (
    SEED,
    BATCH_SIZE,
    DEVICE,
    get_model_info,
    validate_model_type,
    DEFAULT_MODEL_TYPE,
    CROSS_DATASET_TEST_DATASETS,
)
from dataset.dataset import CelebDFDataset
from models.resnet18 import build_model
from util import set_seed
from tqdm import tqdm


def load_ensemble_models(model_type, num_folds=5):
    """
    複数のFoldのモデルを読み込む
    
    Args:
        model_type (str): モデルタイプ ("3ch" または "6ch")
        num_folds (int): Fold数
    
    Returns:
        list: 読み込んだモデルのリスト
    """
    from config import get_celebdf_model_path
    
    models = []
    model_info = get_model_info(model_type)
    in_channels = model_info["in_channels"]
    
    for fold in range(num_folds):
        model_path = get_celebdf_model_path(model_type, f"celebdf_fold{fold}")
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found for fold {fold}: {model_path}")
            continue
        
        model = build_model(in_channels=in_channels, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        models.append(model)
    
    if len(models) == 0:
        print("❌ No models loaded!")
        sys.exit(1)
    
    return models


def ensemble_predict(models, test_loader):
    """
    複数モデルのアンサンブル推論
    
    Args:
        models (list): モデルのリスト
        test_loader: テストデータローダー
    
    Returns:
        dict: 推論結果とメトリクス
    """
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Evaluation", position=0, leave=False)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # 全モデルの予測確率を集計
            ensemble_probs = None
            
            for model_idx, model in enumerate(models):
                out = model(x)
                probs = torch.softmax(out, dim=1)
                
                if ensemble_probs is None:
                    ensemble_probs = probs.cpu().numpy()
                else:
                    ensemble_probs += probs.cpu().numpy()
            
            # 平均化
            ensemble_probs /= len(models)
            
            # 予測クラス
            preds = np.argmax(ensemble_probs, axis=1)
            
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(ensemble_probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # メトリクス計算
    from util import _compute_metrics
    
    metrics = _compute_metrics(all_labels, all_preds, all_probs, num_classes=2)
    
    accuracy = (all_preds == all_labels).mean()
    
    return {
        'accuracy': accuracy,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'precision_weighted': metrics['precision_weighted'],
        'recall_weighted': metrics['recall_weighted'],
        'f1_weighted': metrics['f1_weighted'],
        'roc_auc': metrics['roc_auc'],
        'confusion_matrix': metrics['confusion_matrix'],
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def test_on_dataset(models, dataset_name, dataset_root, model_type, use_residual):
    """
    単一のテストデータセットで評価
    
    対応するデータセット構造：
    - Celeb-DF: dataset_root/real/train, dataset_root/synthesis/train
    - FF++: dataset_root/rsc/raw/real/test, dataset_root/rsc/raw/synthesis/test
    
    Args:
        models (list): モデルのリスト
        dataset_name (str): データセット名
        dataset_root (str): データセットのルートディレクトリ
        model_type (str): モデルタイプ
        use_residual (bool): Residual チャネルを使用するか
    
    Returns:
        dict: 評価結果
    """
    print(f"\n{'='*80}")
    print(f"Testing on {dataset_name} dataset")
    print(f"{'='*80}")
    print(f"Dataset root: {dataset_root}")
    
    # データセット構造の自動検出
    # FF++: rsc/raw/real/test/<videoID>/, rsc/raw/synthesis/test/<videoID>/
    # Celeb-DF: real/train/<videoID>/, synthesis/train/<videoID>/
    
    # まずは dataset_root の直下を確認
    real_base = os.path.join(dataset_root, "real")
    synthesis_base = os.path.join(dataset_root, "synthesis")
    
    # FF++ 構造：real/test, synthesis/test を確認
    if os.path.exists(os.path.join(real_base, "test")) and os.path.exists(os.path.join(synthesis_base, "test")):
        # FF++ 構造（real/test, synthesis/test がある）
        real_dir = os.path.join(real_base, "test")
        synthesis_dir = os.path.join(synthesis_base, "test")
        print(f"Detected: FF++ dataset structure (with real/test, synthesis/test)")
    else:
        # Celeb-DF 構造
        real_dir = real_base
        synthesis_dir = synthesis_base
        print(f"Detected: Celeb-DF dataset structure")
    
    if not os.path.exists(real_dir):
        print(f"❌ Real directory not found: {real_dir}")
        return None
    
    if not os.path.exists(synthesis_dir):
        print(f"❌ Synthesis directory not found: {synthesis_dir}")
        return None
    
    # データセット読み込み
    try:
        test_dataset = CelebDFDataset(
            real_dir=real_dir,
            synthesis_dir=synthesis_dir,
            use_residual=use_residual,
            is_train=False,
            fold=0,
            num_folds=5,
            balance_videos=False
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Test loader batches: {len(test_loader)}")
    
    if len(test_dataset) == 0:
        print(f"❌ No samples found in dataset")
        return None
    
    # アンサンブル推論
    results = ensemble_predict(models, test_loader)
    
    # 結果表示
    print(f"\n{'='*60}")
    print(f"{dataset_name} Test Results")
    print(f"{'='*60}")
    print(f"Accuracy:             {results['accuracy']:.4f}")
    print(f"Precision (macro):    {results['precision_macro']:.4f}")
    print(f"Recall (macro):       {results['recall_macro']:.4f}")
    print(f"F1-Score (macro):     {results['f1_macro']:.4f}")
    print(f"ROC AUC:              {results['roc_auc']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*48}")
    class_names = {0: "Real", 1: "Synthesis"}
    for i in range(2):
        class_name = class_names.get(i, f"Class {i}")
        print(f"{class_name:<12} {results['precision'][i]:<12.4f} "
              f"{results['recall'][i]:<12.4f} {results['f1'][i]:<12.4f}")
    
    # Confusion Matrix
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"{'':>16} Pred: Real  Pred: Synthesis")
    print(f"True: Real       {cm[0, 0]:>6}       {cm[0, 1]:>6}")
    print(f"True: Synthesis  {cm[1, 0]:>6}       {cm[1, 1]:>6}")
    
    # 判別確率をCSVに保存
    probs_csv_path = f"logs/cross_dataset_probabilities/{model_type}/{dataset_name}_probabilities.csv"
    os.makedirs(os.path.dirname(probs_csv_path), exist_ok=True)
    
    prob_data = []
    class_names_mapping = {0: "Real", 1: "Synthesis"}
    
    for idx, (real_prob, synthesis_prob, pred, label) in enumerate(
        zip(results['probabilities'][:, 0], 
            results['probabilities'][:, 1],
            results['predictions'],
            results['labels'])
    ):
        prob_data.append({
            'Sample_ID': idx,
            'Real_Probability': real_prob,
            'Synthesis_Probability': synthesis_prob,
            'Predicted_Class': class_names_mapping[pred],
            'True_Class': class_names_mapping[label],
            'Correct': pred == label
        })
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(probs_csv_path, index=False)
    print(f"\n✓ Probabilities saved to: {probs_csv_path}")
    
    results['dataset_name'] = dataset_name
    results['num_samples'] = len(test_dataset)
    
    return results


def main(model_type=None, dataset_name=None):
    """
    複数データセット上でのクロス評価実行
    
    Args:
        model_type (str, optional): モデルタイプ ("3ch" または "6ch")
        dataset_name (str, optional): データセット名 ("DF", "DFD", "F2F", "FS", "FSfter", "NT", or "all")
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
    
    print(f"\n{'='*80}")
    print(f"Celeb-DF Ensemble Test - Cross Dataset Evaluation")
    print(f"{'='*80}")
    print(f"Model Type: {model_type} ({model_info['description']})")
    print(f"Device: {DEVICE}")
    
    # =========================
    # Seed 固定
    # =========================
    set_seed(SEED)
    
    # =========================
    # 複数モデルの読み込み
    # =========================
    print("\nLoading ensemble models...")
    models = load_ensemble_models(model_type, num_folds=5)
    print(f"✓ Loaded {len(models)} models")
    
    # =========================
    # テストするデータセットを決定
    # =========================
    if dataset_name is None or dataset_name.lower() == "all":
        datasets_to_test = CROSS_DATASET_TEST_DATASETS
    elif dataset_name.upper() in CROSS_DATASET_TEST_DATASETS:
        datasets_to_test = {dataset_name.upper(): CROSS_DATASET_TEST_DATASETS[dataset_name.upper()]}
    else:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"Available datasets: {', '.join(CROSS_DATASET_TEST_DATASETS.keys())}")
        sys.exit(1)
    
    # =========================
    # 各データセットで評価
    # =========================
    results_list = []
    
    for ds_name, ds_root in datasets_to_test.items():
        result = test_on_dataset(models, ds_name, ds_root, model_type, use_residual)
        if result is not None:
            results_list.append(result)
    
    # =========================
    # 結果の集計
    # =========================
    if len(results_list) > 0:
        print(f"\n{'='*80}")
        print(f"Summary - Cross Dataset Evaluation Results")
        print(f"{'='*80}")
        print(f"{'Dataset':<12} {'Samples':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12}")
        print(f"{'-'*80}")
        
        summary_data = []
        for result in results_list:
            print(f"{result['dataset_name']:<12} {result['num_samples']:<10} "
                  f"{result['accuracy']:<12.4f} {result['precision_macro']:<12.4f} "
                  f"{result['recall_macro']:<12.4f} {result['f1_macro']:<12.4f} "
                  f"{result['roc_auc']:<12.4f}")
            
            summary_data.append({
                'Dataset': result['dataset_name'],
                'Samples': result['num_samples'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision_macro'],
                'Recall': result['recall_macro'],
                'F1-Score': result['f1_macro'],
                'ROC AUC': result['roc_auc']
            })
        
        # CSVに保存
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = f"logs/cross_dataset_evaluation_{model_type}.csv"
        os.makedirs("logs", exist_ok=True)
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n✓ Summary results saved to: {summary_csv_path}")
        
        print(f"\n{'='*80}")
        print(f"✓ Cross dataset evaluation completed!")
        print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation with Celeb-DF ensemble models")
    parser.add_argument("--model-type", type=str, default=None, help="Model type: 3ch or 6ch")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset name: DF, DFD, F2F, FS, FSfter, NT, or all")
    
    args = parser.parse_args()
    
    main(model_type=args.model_type, dataset_name=args.dataset)
