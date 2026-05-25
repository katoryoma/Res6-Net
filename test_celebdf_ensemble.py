"""
Celeb-DF Ensemble Test

5つのFold（fold0～fold4）で学習したモデルを使って、
単純平均によるアンサンブルでテストを行うスクリプト

使用方法:
    python test_celebdf_ensemble.py [model_type]
    
    model_type: "3ch" または "6ch" (デフォルト: 6ch)
    
例:
    python test_celebdf_ensemble.py 6ch
    python test_celebdf_ensemble.py 3ch
"""

import os
import sys
import numpy as np
import torch
import pandas as pd

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
)
from dataset.dataset import CelebDFDataset
from models.resnet18 import build_model
from util import set_seed, save_test_log
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
        print(f"✓ Loaded model for fold {fold}")
    
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
        for batch_idx, (x, y) in enumerate(tqdm(test_loader, desc="Ensemble Evaluation", position=0)):
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


def main(model_type=None, num_folds=5):
    """
    Celeb-DFでのアンサンブルテスト実行
    
    Args:
        model_type (str, optional): モデルタイプ ("3ch" または "6ch")
        num_folds (int, optional): Fold数
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
    print(f"Celeb-DF Ensemble Test")
    print(f"{'='*80}")
    print(f"Model Type: {model_type} ({model_info['description']})")
    print(f"Number of Folds: {num_folds}")
    print(f"Device: {DEVICE}")
    
    # =========================
    # Seed 固定
    # =========================
    set_seed(SEED)
    
    # =========================
    # 複数モデルの読み込み
    # =========================
    print("\nLoading ensemble models...")
    models = load_ensemble_models(model_type, num_folds)
    print(f"✓ Loaded {len(models)} models")
    
    # =========================
    # テストデータセット読み込み
    # =========================
    print("\nLoading Celeb-DF test dataset...")
    
    # テスト時は全ビデオを使用
    test_dataset = CelebDFDataset(
        real_dir=os.path.join(CELEBDF_ROOT, "real", "test"),
        synthesis_dir=os.path.join(CELEBDF_ROOT, "synthesis", "test"),
        use_residual=use_residual,
        is_train=False,
        fold=0,
        num_folds=5,
        balance_videos=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Test loader batches: {len(test_loader)}")
    
    # =========================
    # アンサンブル推論実行
    # =========================
    print("\nRunning ensemble evaluation...")
    results = ensemble_predict(models, test_loader)
    
    # =========================
    # 結果表示
    # =========================
    print(f"\n{'='*60}")
    print("Ensemble Test Results")
    print(f"{'='*60}")
    print(f"Accuracy:             {results['accuracy']:.4f}")
    print(f"Precision (macro):    {results['precision_macro']:.4f}")
    print(f"Recall (macro):       {results['recall_macro']:.4f}")
    print(f"F1-Score (macro):     {results['f1_macro']:.4f}")
    print(f"Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"Recall (weighted):    {results['recall_weighted']:.4f}")
    print(f"F1-Score (weighted):  {results['f1_weighted']:.4f}")
    print(f"ROC AUC:              {results['roc_auc']:.4f}")
    print(f"{'='*60}\n")
    
    # Per-class metrics
    print("Per-Class Metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print(f"{'-'*46}")
    class_names = {0: "Real", 1: "Synthesis"}
    for i in range(2):
        class_name = class_names.get(i, f"Class {i}")
        print(f"{class_name:<10} {results['precision'][i]:<12.4f} "
              f"{results['recall'][i]:<12.4f} {results['f1'][i]:<12.4f}")
    
    # Confusion Matrix
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"{'':12} {'Pred: Real':<15} {'Pred: Synthesis':<15}")
    print(f"{'True: Real':<12} {cm[0, 0]:<15} {cm[0, 1]:<15}")
    print(f"{'True: Syn':<12} {cm[1, 0]:<15} {cm[1, 1]:<15}")
    
    # =========================
    # ログ保存
    # =========================
    log_path = os.path.join("logs", "celebdf_ensemble", model_type, "test_log.csv")
    
    # 結果を dict に変換
    log_results = {
        'loss': 0.0,  # アンサンブルではlossは計算しない
        'accuracy': results['accuracy'],
        'per_class_acc': {
            'real': (cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
            'synthesis': (cm[1, 1] / (cm[1, 0] + cm[1, 1])) if (cm[1, 0] + cm[1, 1]) > 0 else 0,
        },
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'precision_macro': results['precision_macro'],
        'recall_macro': results['recall_macro'],
        'f1_macro': results['f1_macro'],
        'precision_weighted': results['precision_weighted'],
        'recall_weighted': results['recall_weighted'],
        'f1_weighted': results['f1_weighted'],
        'roc_auc': results['roc_auc'],
        'confusion_matrix': cm,
    }
    
    save_test_log(log_results, log_path)
    print(f"\n✓ Test results saved to: {log_path}")
    
    # ====================================
    # 判別確率をCSVに保存
    # ====================================
    probs_dir = os.path.dirname(log_path)
    probs_csv_path = os.path.join(probs_dir, "probabilities.csv")
    
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
            'Real_Probability': f"{real_prob:.6f}",
            'Synthesis_Probability': f"{synthesis_prob:.6f}",
            'Predicted_Class': class_names_mapping[pred],
            'True_Class': class_names_mapping[label],
            'Correct': pred == label
        })
    
    prob_df = pd.DataFrame(prob_data)
    prob_df.to_csv(probs_csv_path, index=False)
    print(f"✓ Probabilities saved to: {probs_csv_path}")
    
    print(f"\n{'='*60}")
    print(f"Ensemble test completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    num_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    main(model_type, num_folds)
