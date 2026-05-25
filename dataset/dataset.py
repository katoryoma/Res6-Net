import os
from PIL import Image

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset.transforms import get_geo_transforms
from config import IMAGE_SIZE

class ImageDataset(Dataset):
    """
    画像分類用データセット
    
    Args:
        root_dir (str): データセットのルートディレクトリ
        use_residual (bool): residual チャネルを使用するか（デフォルト: True）
                            True: 6チャネル（RGB + Residual）
                            False: 3チャネル（RGBのみ）
    """

    def __init__(self, root_dir, use_residual=True):

        self.root_dir = root_dir
        self.use_residual = use_residual
        self.current_epoch = 0  # 現在のエポック

        self.samples = []

        classes = ["real", "swap"]

        for label, cls_name in enumerate(classes):

            cls_dir = os.path.join(root_dir, cls_name)

            for folder_name in os.listdir(cls_dir):

                folder_path = os.path.join(cls_dir, folder_name)

                for img_name in os.listdir(folder_path):

                    img_path = os.path.join(folder_path, img_name)

                    self.samples.append((img_path, label))

        self.to_tensor = transforms.ToTensor()

        self.geo_aug = get_geo_transforms()

    def set_epoch(self, epoch):
        """
        現在のエポック情報を設定（ランダムシードの変更に使用）
        
        Args:
            epoch (int): エポック番号
        """
        self.current_epoch = epoch

    def __len__(self):
        return len(self.samples)

    def create_residual(self, x):

        # x: [3, IMAGE_SIZE, IMAGE_SIZE]
        
        # 入力サイズを動的に取得
        size = x.shape[-1]

        # =========================
        # size -> 128
        # =========================
        down = F.interpolate(
            x.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )

        # =========================
        # 128 -> size
        # =========================
        up = F.interpolate(
            down,
            size=(size, size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # =========================
        # 差分
        # =========================
        residual = torch.abs(x - up)

        # =========================
        # log増幅
        # =========================
        residual = torch.log1p(residual * 255)

        # =========================
        # 正規化
        # =========================
        residual = residual / residual.max()

        return residual

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        # 画像の短辺を IMAGE_SIZE にリサイズ（アスペクト比維持）
        width, height = img.size
        if width < height:
            new_width = IMAGE_SIZE
            new_height = int(IMAGE_SIZE * height / width)
        else:
            new_height = IMAGE_SIZE
            new_width = int(IMAGE_SIZE * width / height)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # エポックごとに異なる位置から切り出す（ランダムクロップ）
        # エポック＋インデックスでシードを設定し、確定的だが異なる切り出し位置を作成
        rng = torch.Generator()
        rng.manual_seed(self.current_epoch * 10000 + idx)
        
        if new_width > IMAGE_SIZE:
            left = int(torch.randint(0, new_width - IMAGE_SIZE + 1, (1,), generator=rng).item())
        else:
            left = 0
            
        if new_height > IMAGE_SIZE:
            top = int(torch.randint(0, new_height - IMAGE_SIZE + 1, (1,), generator=rng).item())
        else:
            top = 0
        
        right = left + IMAGE_SIZE
        bottom = top + IMAGE_SIZE
        img = img.crop((left, top, right, bottom))

        # 幾何学変換
        if self.geo_aug is not None:
            img = self.geo_aug(img)
            
        # tensor化
        rgb = self.to_tensor(img)

        # residual処理
        if self.use_residual == True:
            # 6チャネル（RGB + Residual）
            residual = self.create_residual(rgb)
            x = torch.cat([rgb, residual], dim=0)
        elif self.use_residual == "residual_only":
            # 3チャネル（Residualのみ）
            x = self.create_residual(rgb)
        else:
            # 3チャネル（RGBのみ）
            x = rgb

        return x, label


# =========================
# 互換性のためのエイリアス
# =========================
SixChannelDataset = ImageDataset


# =========================
# Celeb-DF Dataset
# =========================
class CelebDFDataset(Dataset):
    """
    Celeb-DF形式のデータセット（動画フレームベース）
    
    データ構造:
    CELEBDF_ROOT/
    ├── real/
    │   ├── train/
    │   │   ├── <videoID>/
    │   │   │   ├── frame_000000.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── test/
    │       ├── <videoID>/
    │       │   └── ...
    │       └── ...
    └── synthesis/
        ├── train/
        │   ├── <videoID>/
        │   │   └── ...
        │   └── ...
        └── test/
            └── ...
    
    Args:
        real_dir (str): real クラスのディレクトリ
        synthesis_dir (str): synthesis クラスのディレクトリ
        use_residual (bool): residual チャネルを使用するか（デフォルト: True）
        is_train (bool): 訓練モードかテストモードか（デフォルト: True）
        val_split (float): 訓練データの何%を検証用に使用するか（デフォルト: 0.2）
        fold (int): Cross-Validationの折番号（デフォルト: 0）
        balance_videos (bool): 訓練時にvideo数をrealに合わせるか（デフォルト: True）
    """
    
    def __init__(self, real_dir, synthesis_dir, use_residual=True, is_train=True, 
                 val_split=0.2, fold=0, num_folds=5, balance_videos=True):
        
        self.real_dir = real_dir
        self.synthesis_dir = synthesis_dir
        self.use_residual = use_residual
        self.is_train = is_train
        self.val_split = val_split
        self.fold = fold
        self.num_folds = num_folds
        self.current_epoch = 0
        
        # テストセット（"test" を含むディレクトリ）の場合、balance_videos を無効化
        # テストセットは balanced_synthesis.txt を必要としない
        is_test_dataset = "test" in real_dir or "test" in synthesis_dir
        if is_test_dataset:
            self.balance_videos = False
        else:
            self.balance_videos = balance_videos
        
        self.samples = []
        self.video_groups = {"real": [], "synthesis": []}  # ビデオIDのリスト
        
        # balanced_synthesis.txt から固定の synthesis 動画 ID を読み込む
        self.fixed_balanced_synthesis_videos = self._load_balanced_synthesis_list()
        
        # ビデオを収集
        self._collect_videos()
        
        # Cross-Validationで訓練/検証を分割
        self._split_videos()
        
        # 訓練時：初期化時に1回だけ合成動画を real に合わせる
        if self.balance_videos:
            self._balance_videos_once()
            # balance 後は samples が既に構成されている
        else:
            # balance_videos=False：通常通りサンプルを生成
            self._create_samples()
        
        self.to_tensor = transforms.ToTensor()
        self.geo_aug = get_geo_transforms()
    
    def _load_balanced_synthesis_list(self):
        """
        balanced_synthesis.txt から固定の synthesis 動画 ID リストを読み込む
        
        Returns:
            set: balanced_synthesis.txt に記載された動画 ID のセット
        """
        # スクリプトの実行ディレクトリからの相対パスで balanced_synthesis.txt を探す
        balanced_list_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "balanced_synthesis.txt")
        
        # ない場合は現在のディレクトリで探す
        if not os.path.exists(balanced_list_path):
            balanced_list_path = "balanced_synthesis.txt"
        
        if not os.path.exists(balanced_list_path):
            print(f"Warning: {balanced_list_path} not found. balance_videos will be disabled.")
            return set()
        
        video_ids = set()
        try:
            with open(balanced_list_path, "r") as f:
                for line in f:
                    video_id = line.strip()
                    if video_id:
                        video_ids.add(video_id)
            print(f"Loaded {len(video_ids)} fixed synthesis videos from {balanced_list_path}")
        except Exception as e:
            print(f"Error reading {balanced_list_path}: {e}")
        
        return video_ids
    
    def _collect_videos(self):
        """ビデオIDを収集"""
        class_dirs = {
            "real": self.real_dir,
            "synthesis": self.synthesis_dir
        }
        
        for class_name, class_dir in class_dirs.items():
            
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist")
                continue
            
            for video_id in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_id)
                
                if os.path.isdir(video_path):
                    # フレームファイルが存在するかチェック
                    frames = [f for f in os.listdir(video_path) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))]
                    
                    if len(frames) > 0:
                        self.video_groups[class_name].append(video_id)
    
    def _split_videos(self):
        """
        K-fold Cross-Validationで訓練/検証ビデオを分割
        
        テストセット（real_dir/synthesis_dir に "test" を含む）の場合：
          - is_train=False の時は分割をせずすべてのビデオを使用
        
        訓練セット（real_dir/synthesis_dir に "test" を含まない）の場合：
          - is_train=True: 訓練データ（4 fold）を train_videos に
          - is_train=False: 検証データ（1 fold）を val_videos に
        
        例）5-fold の場合（訓練セット）：
            fold 0: val=[A], train=[B,C,D,E]
            fold 1: val=[B], train=[A,C,D,E]
            fold 2: val=[C], train=[A,B,D,E]
            ...
        """
        import random
        from collections import defaultdict
        
        self.train_videos = defaultdict(list)
        self.val_videos = defaultdict(list)
        
        # テストセットかどうかを判定
        is_test_dataset = "test" in self.real_dir
        
        # テストセット かつ テストモード（is_train=False）の場合：分割をせずすべてのビデオを使用
        if is_test_dataset and not self.is_train:
            for class_name in ["real", "synthesis"]:
                self.val_videos[class_name] = self.video_groups[class_name].copy()
            return
        
        # 訓練セットの場合：常に Cross-Validation 分割を適用
        # 固定シード（fold に依存しない）でシャッフル
        rng = random.Random(42)
        
        for class_name in ["real", "synthesis"]:
            videos = self.video_groups[class_name].copy()
            rng.shuffle(videos)
            
            # num_folds に分割
            fold_size = len(videos) // self.num_folds
            folds = []
            for i in range(self.num_folds):
                start_idx = i * fold_size
                if i == self.num_folds - 1:
                    # 最後のフォールドは余りを含める
                    folds.append(videos[start_idx:])
                else:
                    folds.append(videos[start_idx:start_idx + fold_size])
            
            # 現在の fold を検証用に、他を訓練用に
            val_vids = folds[self.fold]
            train_vids = []
            for i, f in enumerate(folds):
                if i != self.fold:
                    train_vids.extend(f)
            
            self.train_videos[class_name] = train_vids
            self.val_videos[class_name] = val_vids
    
    def _create_samples(self):
        """フレームのパスとラベルを作成"""
        self.samples = []
        
        if self.is_train:
            videos_dict = self.train_videos
            class_dirs = {
                "real": self.real_dir,
                "synthesis": self.synthesis_dir
            }
        else:
            videos_dict = self.val_videos
            class_dirs = {
                "real": self.real_dir,
                "synthesis": self.synthesis_dir
            }
        
        for class_idx, class_name in enumerate(["real", "synthesis"]):
            video_ids = videos_dict[class_name]
            class_dir = class_dirs[class_name]
            
            for video_id in video_ids:
                video_path = os.path.join(class_dir, video_id)
                
                if os.path.exists(video_path):
                    # フレームをソートして順序を確保
                    frames = sorted([f for f in os.listdir(video_path) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
                    
                    for frame_name in frames:
                        frame_path = os.path.join(video_path, frame_name)
                        self.samples.append((frame_path, class_idx))
    
    def set_epoch(self, epoch):
        """エポック情報を設定"""
        self.current_epoch = epoch
        # fold決定時に1回だけ balance されているため、ここでは何もしない
    
    def _balance_videos_once(self):
        """
        初期化時に1回だけ、balanced_synthesis.txt に記載された固定動画リストを使用
        すべての fold で同じ synthesis 動画セットを使用することで CV を成り立たせる
        """
        import random
        
        if self.is_train:
            target_video_group = self.train_videos
            mode_label = "Train"

        else:
            target_video_group = self.val_videos
            mode_label = "Val"

        # balanced_synthesis.txt から読み込んだ固定リストを使用
        if not self.fixed_balanced_synthesis_videos:
            print(f"Warning: No balanced synthesis videos loaded. Using all available videos.")
            selected_syn_videos = target_video_group["synthesis"]
        else:
            # fixed_balanced_synthesis_videos に含まれる動画のみをフィルタ
            selected_syn_videos = [
                vid for vid in target_video_group["synthesis"] 
                if vid in self.fixed_balanced_synthesis_videos
            ]
            print(f"[Fold {self.fold}] Using fixed balanced synthesis videos: "
                  f"selected {len(selected_syn_videos)} videos "
                  f"(from balanced_synthesis.txt)")
        
        real_videos = target_video_group["real"]
        
        # 選択された動画のフレームのみを再構成
        self.samples = []
        
        # Realフレームを追加
        for video_id in real_videos:
            video_path = os.path.join(self.real_dir, video_id)
            if os.path.exists(video_path):
                frames = sorted([f for f in os.listdir(video_path) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
                for frame_name in frames:
                    frame_path = os.path.join(video_path, frame_name)
                    self.samples.append((frame_path, 0))
        
        # Synthesisフレームを追加（選択されたビデオのみ）
        for video_id in selected_syn_videos:
            video_path = os.path.join(self.synthesis_dir, video_id)
            if os.path.exists(video_path):
                frames = sorted([f for f in os.listdir(video_path) 
                               if f.endswith(('.png', '.jpg', '.jpeg'))])
                for frame_name in frames:
                    frame_path = os.path.join(video_path, frame_name)
                    self.samples.append((frame_path, 1))
        
        # シャッフル（fold ごとに固定されたシード）
        rng = random.Random(42 + self.fold)
        rng.shuffle(self.samples)
    
    def _balance_samples(self):
        """合成動画の数をrealに合わせるようにサンプルを再構成（非推奨：互換性のため残す）"""
        # 新しい動画単位の均衡化メソッドに委譲
        self._balance_samples_by_videos()
    
    def __len__(self):
        return len(self.samples)
    
    def create_residual(self, x):
        """Residual チャネルを作成"""
        # x: [3, IMAGE_SIZE, IMAGE_SIZE]
        
        # 入力サイズを動的に取得
        size = x.shape[-1]
        
        # size -> 128
        down = F.interpolate(
            x.unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        )
        
        # 128 -> size
        up = F.interpolate(
            down,
            size=(size, size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        # 差分
        residual = torch.abs(x - up)
        
        # log増幅
        residual = torch.log1p(residual * 255)
        
        # 正規化
        residual = residual / residual.max()
        
        return residual
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        # 画像の短辺を IMAGE_SIZE にリサイズ（アスペクト比維持）
        width, height = img.size
        if width < height:
            new_width = IMAGE_SIZE
            new_height = int(IMAGE_SIZE * height / width)
        else:
            new_height = IMAGE_SIZE
            new_width = int(IMAGE_SIZE * width / height)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # エポックごとに異なる位置から切り出す
        rng = torch.Generator()
        rng.manual_seed(self.current_epoch * 10000 + idx)
        
        if new_width > IMAGE_SIZE:
            left = int(torch.randint(
                0, new_width - IMAGE_SIZE + 1, (1,), generator=rng
            ).item())
        else:
            left = 0
        
        if new_height > IMAGE_SIZE:
            top = int(torch.randint(
                0, new_height - IMAGE_SIZE + 1, (1,), generator=rng
            ).item())
        else:
            top = 0
        
        right = left + IMAGE_SIZE
        bottom = top + IMAGE_SIZE
        img = img.crop((left, top, right, bottom))
        
        # 幾何学変換
        if self.geo_aug is not None:
            img = self.geo_aug(img)
        
        # tensor化
        rgb = self.to_tensor(img)
        
        # residual処理
        if self.use_residual == True:
            # 6チャネル（RGB + Residual）
            residual = self.create_residual(rgb)
            x = torch.cat([rgb, residual], dim=0)
        elif self.use_residual == "residual_only":
            # 3チャネル（Residualのみ）
            x = self.create_residual(rgb)
        else:
            # 3チャネル（RGBのみ）
            x = rgb
        
        return x, label