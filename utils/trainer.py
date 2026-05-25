import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader

from dataset.dataset import SixChannelDataset
from models.resnet18 import build_model
from .util import accuracy, set_seed, save_model


class Trainer:

    def __init__(self, train_loader, val_loader, model, device):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.best_acc = 0.0
        self.patience_counter = 0  # Early stopping カウンター

    # =========================
    # train 1 epoch
    # =========================
    def train_one_epoch(self):

        self.model.train()

        total_loss = 0
        total_acc = 0

        for x, y in tqdm(self.train_loader, desc="Training", leave=False):

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            out = self.model(x)
            loss = self.criterion(out, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(out, y)

        return total_loss / len(self.train_loader), total_acc / len(self.train_loader)

    # =========================
    # validation
    # =========================
    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0
        total_acc = 0

        for x, y in tqdm(self.val_loader, desc="Validation", leave=False):

            x = x.to(self.device)
            y = y.to(self.device)

            out = self.model(x)
            loss = self.criterion(out, y)

            total_loss += loss.item()
            total_acc += accuracy(out, y)

        return total_loss / len(self.val_loader), total_acc / len(self.val_loader)

    # =========================
    # full training loop with early stopping
    # =========================
    def fit(self, epochs=20, save_path="best_model.pth", log_path=None, patience=10):
        """
        Early stopping 機能付きトレーニング
        
        Args:
            epochs (int): 最大エポック数
            save_path (str): ベストモデルの保存先
            log_path (str): ログの保存先
            patience (int): val_acc が向上しない連続エポック数の上限（デフォルト: 10）
        """

        for epoch in tqdm(range(1, epochs + 1), desc="Epochs", position=0):

            # エポック開始時にデータセットのエポック情報を更新
            # これにより、各エポックで異なる位置から画像を切り出す
            self.train_loader.dataset.set_epoch(epoch - 1)

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            tqdm.write(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # =========================
            # save best model and check early stopping
            # =========================
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.patience_counter = 0  # リセット
                save_model(self.model, save_path)
                tqdm.write(f"✔ Saved best model (acc={val_acc:.4f})")
            else:
                self.patience_counter += 1
                tqdm.write(f"Val Acc did not improve. Patience: {self.patience_counter}/{patience}")
                
                # Early stopping
                if self.patience_counter >= patience:
                    tqdm.write(f"⛔ Early stopping triggered! No improvement for {patience} consecutive epochs.")
                    break
            
            # =========================
            # ログ保存
            # =========================
            if log_path is not None:
                self._save_log(epoch, train_loss, train_acc, val_loss, val_acc, log_path)

    def _save_log(self, epoch, train_loss, train_acc, val_loss, val_acc, log_path):
        """ログをCSVに保存"""
        import csv
        import os
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        file_exists = os.path.exists(log_path)
        
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # ヘッダーを書き込み
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])
            
            # ログを書き込み
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])


# =========================
# run
# =========================
def main():

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Dataset
    # =========================
    train_dataset = SixChannelDataset(
        root_dir=r"!YOUR_DATAPATH!\my_data\image\raw\train"
    )

    val_dataset = SixChannelDataset(
        root_dir=r"!YOUR_DATAPATH!\my_data\image\raw\validation",
        geo_aug=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # =========================
    # Model (6ch)
    # =========================
    model = build_model(in_channels=6)

    # =========================
    # Trainer
    # =========================
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        device=device
    )

    trainer.fit(epochs=20)


if __name__ == "__main__":
    main()
