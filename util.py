# util.py

import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm


# =========================
# seed固定
# =========================
def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# エイリアス (trainer.py との互換性)
set_seed = seed_everything


# =========================
# accuracy計算
# =========================
def calculate_accuracy(outputs, labels):

    preds = torch.argmax(outputs, dim=1)

    correct = (preds == labels).sum().item()

    acc = correct / labels.size(0)

    return acc


# エイリアス (trainer.py との互換性)
accuracy = calculate_accuracy


# =========================
# モデル保存
# =========================
def save_model(model, save_path):
    """モデルのstate_dictのみを保存"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved -> {save_path}")


# =========================
# checkpoint保存
# =========================
def save_checkpoint(model, optimizer, epoch, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, save_path)

    print(f"Checkpoint saved -> {save_path}")


# =========================
# checkpoint読み込み
# =========================
def load_checkpoint(model, optimizer, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]

    print(f"Checkpoint loaded -> {checkpoint_path}")

    return model, optimizer, epoch


# =========================
# メトリクス計算（scikit-learn なし）
# =========================
def _confusion_matrix(y_true, y_pred, num_classes=2):
    """NumPy で混同行列を計算"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm


def _compute_roc_auc_binary(y_true, y_scores):
    """Binary ROC AUC"""
    sorted_indices = np.argsort(-y_scores)
    sorted_y = y_true[sorted_indices]
    
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tp = 0
    fp = 0
    auc = 0.0
    
    for idx in sorted_indices:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    
    return auc / (n_pos * n_neg)


def _compute_roc_auc_multiclass(y_true, y_probs, num_classes):
    """Multiclass ROC AUC (OVR)"""
    aucs = []
    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
            auc = _compute_roc_auc_binary(y_binary, y_probs[:, i])
            aucs.append(auc)
    return np.mean(aucs) if aucs else 0.5


def _compute_metrics(y_true, y_pred, y_probs, num_classes=2):
    """NumPy でメトリクスを計算"""
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    
    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        precision[i] = p
        recall[i] = r
        f1[i] = f
    
    # Macro averages
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    # Weighted averages
    weights = np.array([cm[i, :].sum() for i in range(num_classes)])
    weights = weights / weights.sum()
    precision_weighted = (precision * weights).sum()
    recall_weighted = (recall * weights).sum()
    f1_weighted = (f1 * weights).sum()
    
    # ROC AUC
    if num_classes == 2:
        # Binary classification: use probs of positive class
        roc_auc = _compute_roc_auc_binary(y_true, y_probs[:, 1])
    else:
        roc_auc = _compute_roc_auc_multiclass(y_true, y_probs, num_classes)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }


def evaluate(model, data_loader, device, num_classes=2):
    """
    モデルをテストセットで詳細評価
    テスト時は画像ごとに異なる切り出し位置を使用
    
    Args:
        model: 評価するモデル
        data_loader: テストデータローダー
        device: デバイス
        num_classes: クラス数
    
    Returns:
        dict: 詳細なメトリクス情報
    """
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_acc = 0.0
    
    # クラス別の正解数とサンプル数
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    # 予測結果と真値を収集
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm(data_loader, desc="Evaluating", position=0)):
            
            # バッチごとに異なるエポック番号を設定
            # これにより、各バッチで異なる切り出し位置が使われる
            data_loader.dataset.set_epoch(batch_idx)
            
            x = x.to(device)
            y = y.to(device)
            
            out = model(x)
            loss = criterion(out, y)
            
            total_loss += loss.item()
            total_acc += accuracy(out, y)
            
            # クラス別精度を計算
            preds = torch.argmax(out, dim=1)
            probs = torch.softmax(out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for i in range(num_classes):
                mask = (y == i)
                class_correct[i] += (preds[mask] == y[mask]).sum().item()
                class_total[i] += mask.sum().item()
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
    
    # クラス別精度
    per_class_acc = {}
    class_names = {0: "real", 1: "swap"}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[class_names.get(i, f"class_{i}")] = class_correct[i] / class_total[i]
    
    # 詳細メトリクス計算（NumPy を使用）
    metrics = _compute_metrics(all_labels, all_preds, all_probs, num_classes)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_acc,
        'per_class_acc': per_class_acc,
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



# =========================
# テストログ保存
# =========================
def save_test_log(results, log_path):
    """
    テストログをCSVに保存
    
    Args:
        results (dict): evaluate()の返却値
        log_path (str): 保存先パス
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    class_names = {0: "real", 1: "swap"}
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # ===== 基本メトリクス =====
        writer.writerow(['=== Basic Metrics ==='])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Test Loss', results['loss']])
        writer.writerow(['Test Accuracy', results['accuracy']])
        writer.writerow(['ROC AUC', results['roc_auc']])
        
        # ===== クラス別精度 =====
        writer.writerow([''])
        writer.writerow(['=== Per-Class Accuracy ==='])
        writer.writerow(['Class', 'Accuracy'])
        for class_name, acc in results['per_class_acc'].items():
            writer.writerow([class_name, acc])
        
        # ===== Precision, Recall, F1 (Per-Class) =====
        writer.writerow([''])
        writer.writerow(['=== Per-Class Metrics ==='])
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score'])
        for i in range(2):
            class_name = class_names.get(i, f"class_{i}")
            writer.writerow([
                class_name,
                results['precision'][i],
                results['recall'][i],
                results['f1'][i]
            ])
        
        # ===== マクロ平均 =====
        writer.writerow([''])
        writer.writerow(['=== Macro Averages ==='])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', results['precision_macro']])
        writer.writerow(['Recall', results['recall_macro']])
        writer.writerow(['F1-Score', results['f1_macro']])
        
        # ===== ウェイト平均 =====
        writer.writerow([''])
        writer.writerow(['=== Weighted Averages ==='])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', results['precision_weighted']])
        writer.writerow(['Recall', results['recall_weighted']])
        writer.writerow(['F1-Score', results['f1_weighted']])
        
        # ===== 混同行列 =====
        writer.writerow([''])
        writer.writerow(['=== Confusion Matrix ==='])
        writer.writerow(['', 'Pred: real', 'Pred: swap'])
        cm = results['confusion_matrix']
        writer.writerow(['True: real', cm[0, 0], cm[0, 1]])
        writer.writerow(['True: swap', cm[1, 0], cm[1, 1]])