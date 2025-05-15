import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
import warnings

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, model, config):
        """
        Args:
            model: 待训练的PyTorch模型
            config: 包含训练配置的字典，必须包含：
                - device: torch.device
                - epochs: int
                - optimizer: torch.optim
                - loss_fn: 损失函数
                - train_loader: DataLoader
                - val_loader: DataLoader
                - save_dir: str (模型保存路径)
                - early_stop: int (早停轮数)
                - grad_clip: float (梯度裁剪阈值)
                - lr_scheduler: 学习率调度器 (可选)
                - metric: 主评估指标 ('acc' 或 'f1')
        """
        self.model = model.to(config['device'])
        self.config = config
        self.best_metric = 0
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }

    def train_epoch(self):
        self.model.train()
        total_loss, total_samples = 0, 0
        all_preds, all_labels = [], []

        with tqdm(self.config['train_loader'], desc='Training') as pbar:
            for x, y in pbar:
                x = x.to(self.config['device'])
                y = y.to(self.config['device'])

                # 前向传播
                outputs = self.model(x)
                loss = self.config['loss_fn'](outputs, y)

                # 反向传播
                self.config['optimizer'].zero_grad()
                loss.backward()

                # 梯度裁剪
                if self.config['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )

                self.config['optimizer'].step()

                # 记录指标
                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                pbar.set_postfix({
                    'loss': total_loss / total_samples,
                    'acc': accuracy_score(all_labels, all_preds)
                })

        # 计算epoch指标
        epoch_loss = total_loss / total_samples
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        self.history['train_f1'].append(epoch_f1)

        return epoch_loss, epoch_acc, epoch_f1

    def validate(self):
        self.model.eval()
        total_loss, total_samples = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in self.config['val_loader']:
                x = x.to(self.config['device'])
                y = y.to(self.config['device'])

                outputs = self.model(x)
                loss = self.config['loss_fn'](outputs, y)

                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 计算验证指标
        val_loss = total_loss / total_samples
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)

        return val_loss, val_acc, val_f1

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.config['optimizer'].state_dict(),
            'best_metric': self.best_metric
        }

        if is_best:
            torch.save(state, os.path.join(
                self.config['save_dir'], 'best_model.pth'))

        torch.save(state, os.path.join(
            self.config['save_dir'], 'last_checkpoint.pth'))

    def train(self):
        for epoch in range(1, self.config['epochs'] + 1):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch}/{self.config['epochs']}")

            # 训练与验证
            train_loss, train_acc, train_f1 = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            # 学习率调度
            if self.config.get('lr_scheduler'):
                if isinstance(self.config['lr_scheduler'], torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.config['lr_scheduler'].step(val_loss)
                else:
                    self.config['lr_scheduler'].step()

            # 打印指标
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

            # 早停与模型保存
            current_metric = val_f1 if self.config['metric'] == 'f1' else val_acc

            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.epochs_no_improve = 0
                self.save_checkpoint(is_best=True)
                print(f"New best model! ({self.best_metric:.4f})")
            else:
                self.epochs_no_improve += 1
                self.save_checkpoint()
                if self.epochs_no_improve >= self.config['early_stop']:
                    print(f"No improvement for {self.config['early_stop']} epochs. Early stopping!")
                    break

        return self.history
