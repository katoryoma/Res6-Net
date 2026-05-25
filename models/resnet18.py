import torch
import torch.nn as nn
from torchvision import models


def build_model(in_channels=6, num_classes=2):
    """
    ResNet18 モデルを構築
    
    Args:
        in_channels (int): 入力チャネル数（デフォルト: 6 for RGB + Residual）
        num_classes (int): 出力クラス数（デフォルト: 2 for real / swap）
    
    Returns:
        model: ResNet18 モデル
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 事前学習済みモデルResNet18
    model = models.resnet18(weights="IMAGENET1K_V1")

    old_conv = model.conv1

    # 入力チャネル変更
    model.conv1 = nn.Conv2d(
        in_channels,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    # 重みコピー
    with torch.no_grad():
        if in_channels >= 3:
            model.conv1.weight[:, :3] = old_conv.weight
            if in_channels >= 6:
                model.conv1.weight[:, 3:6] = old_conv.weight

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    model = model.to(device)

    return model

