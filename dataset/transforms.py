import torchvision.transforms as T

def get_geo_transforms(image_size=512):
    """
    幾何学変換のみのデータ拡張
    Deepfake検出向け（形状保持重視）
    """

    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),

        T.RandomRotation(degrees=10),

        T.RandomResizedCrop(
            size=image_size,
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05)
        )
    ])