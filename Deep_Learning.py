import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
import signal
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import cv2
from sklearn.metrics import jaccard_score

# 1. CONFIG
class Config:
    BASE_DIR = r"K:\UNI\MAHBOD\OpenEarthMap\OpenEarthMap_wo_xBD"
    IMG_SIZE = 256
    NUM_CLASSES = 5
    BATCH_SIZE = 8
    LR = 1e-4
    EPOCHS = 50
    
    # Checkpoint settings
    CKPT_FREQ = 2
    KEEP_CKPT = 3
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    CKPT_DIR = f'checkpoints_{TIMESTAMP}'
    LOG_DIR = f'logs_{TIMESTAMP}'
    VIZ_DIR = f'viz_{TIMESTAMP}'
    
    Path(CKPT_DIR).mkdir(exist_ok=True)
    Path(LOG_DIR).mkdir(exist_ok=True)
    Path(VIZ_DIR).mkdir(exist_ok=True)

# 2. DATASET
class DamageDataset(Dataset):
    def __init__(self, file_list_path, transform=None, max_samples=500):
        with open(file_list_path, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        self.files = []
        for fname in files:
            region = '_'.join(fname.split('_')[:-1])
            img_path = os.path.join(Config.BASE_DIR, region, 'images', fname)
            mask_path = os.path.join(Config.BASE_DIR, region, 'labels', fname)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.files.append(fname)
        
        if len(self.files) > max_samples:
            self.files = random.sample(self.files, max_samples)
        
        self.transform = transform
        print(f"Loaded {len(self.files)} samples")

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        region = '_'.join(fname.split('_')[:-1])
        
        img_path = os.path.join(Config.BASE_DIR, region, 'images', fname)
        mask_path = os.path.join(Config.BASE_DIR, region, 'labels', fname)
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.int64)
            mask = np.clip(mask, 0, Config.NUM_CLASSES - 1)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask'].long()
            else:
                image = torch.from_numpy(image).float().permute(2, 0, 1)
                mask = torch.from_numpy(mask).long()
                
            return image, mask, fname
            
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            dummy_img = torch.rand(3, Config.IMG_SIZE, Config.IMG_SIZE)
            dummy_mask = torch.zeros(Config.IMG_SIZE, Config.IMG_SIZE, dtype=torch.long)
            return dummy_img, dummy_mask, 'error'

# 3. MODEL
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=Config.NUM_CLASSES, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        for feature in features:
            self.downs.append(self._block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(self._block(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skips = []
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i//2]
            
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear')
                
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i+1](x)

        return self.final_conv(x)

# 4. VISUALIZER
class Visualizer:
    COLORS = np.array([
        [0, 0, 0],       # Class 0
        [0, 0, 255],     # Class 1  
        [255, 255, 0],   # Class 2
        [0, 255, 0],     # Class 3
        [255, 0, 0],     # Class 4
    ], dtype=np.uint8)
    
    @classmethod
    def mask_to_rgb(cls, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(Config.NUM_CLASSES):
            rgb[mask == i] = cls.COLORS[i]
            
        return rgb
    
    @classmethod
    def create_grid(cls, images, true_masks, pred_masks, num_samples=3):
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        titles = ['Input Image', 'Ground Truth', 'Prediction', 'Comparison']
        
        for i in range(num_samples):
            # Input
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(titles[0], fontweight='bold')
            axes[i, 0].axis('off')
            
            # Ground Truth
            true_rgb = cls.mask_to_rgb(true_masks[i])
            axes[i, 1].imshow(true_rgb)
            axes[i, 1].set_title(titles[1], fontweight='bold', color='green')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_rgb = cls.mask_to_rgb(pred_masks[i])
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title(titles[2], fontweight='bold', color='blue')
            axes[i, 2].axis('off')
            
            # Comparison
            comp = np.hstack([true_rgb, pred_rgb])
            axes[i, 3].imshow(comp)
            axes[i, 3].set_title(titles[3], fontweight='bold', color='red')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        return fig
    
    @classmethod
    def fig_to_tensor(cls, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.transpose(2, 0, 1)
    
    @classmethod
    def save_fig(cls, fig, epoch, prefix="val"):
        path = os.path.join(Config.VIZ_DIR, f"{prefix}_epoch_{epoch:03d}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

# 5. TRAINER
class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.writer = SummaryWriter(Config.LOG_DIR)
        
        self.model = UNET().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LR)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler('cuda') if self.device == 'cuda' else None
        
        self.train_loader, self.val_loader = self._get_loaders()
        self.state_file = os.path.join(Config.CKPT_DIR, 'state.pth')
        
        self.epoch = 1
        self.best_iou = 0
        self.train_loss = []
        self.val_iou = []
        self.ckpt_files = []
        
        self._setup_handlers()
        self._setup_logs()

    def _get_loaders(self):
        transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ])
        
        train_ds = DamageDataset(os.path.join(Config.BASE_DIR, 'train.txt'), transform)
        val_ds = DamageDataset(os.path.join(Config.BASE_DIR, 'val.txt'), transform)
        
        return (
            DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True),
            DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False)
        )

    def _setup_handlers(self):
        def handler(sig, frame):
            print("Saving state before exit...")
            self.save_state()
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)

    def _setup_logs(self):
        self.log_file = os.path.join(Config.VIZ_DIR, "progress.md")
        with open(self.log_file, 'w') as f:
            f.write("# Training Progress\n\n")
            f.write("| Epoch | Loss | IoU | Best |\n")
            f.write("|-------|------|-----|------|\n")

    def _update_log(self, epoch, loss, iou):
        best = "★" if iou > self.best_iou else ""
        with open(self.log_file, 'a') as f:
            f.write(f"| {epoch} | {loss:.4f} | {iou:.4f} | {best} |\n")

    def _clean_ckpts(self):
        if len(self.ckpt_files) > Config.KEEP_CKPT:
            for f in self.ckpt_files[:-Config.KEEP_CKPT]:
                if os.path.exists(f):
                    os.remove(f)
            self.ckpt_files = self.ckpt_files[-Config.KEEP_CKPT:]

    def save_state(self):
        state = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'best_iou': self.best_iou,
            'train_loss': self.train_loss,
            'val_iou': self.val_iou,
        }
        torch.save(state, self.state_file)

    def load_state(self):
        if os.path.exists(self.state_file):
            state = torch.load(self.state_file)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            if self.scaler and state['scaler']:
                self.scaler.load_state_dict(state['scaler'])
            self.epoch = state['epoch'] + 1
            self.best_iou = state['best_iou']
            self.train_loss = state['train_loss']
            self.val_iou = state['val_iou']
            print(f"Resumed from epoch {self.epoch}")
            return True
        return False

    def save_ckpt(self, name, best=False):
        ckpt = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'best_iou': self.best_iou,
        }
        path = os.path.join(Config.CKPT_DIR, f"{name}.pth")
        torch.save(ckpt, path)
        
        if name.startswith('epoch_'):
            self.ckpt_files.append(path)
            self._clean_ckpts()
        
        if best:
            print(f"New best model! IoU: {self.best_iou:.4f}")

    def calc_iou(self, preds, masks):
        preds_flat = preds.cpu().numpy().flatten()
        masks_flat = masks.cpu().numpy().flatten()
        return jaccard_score(masks_flat, preds_flat, average='macro', 
                           labels=range(Config.NUM_CLASSES), zero_division=0)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for images, masks, _ in bar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if self.device == 'cuda' else 'cpu'):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        all_preds, all_masks = [], []
        samples = []
        
        with torch.no_grad():
            for i, (images, masks, names) in enumerate(self.val_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(1)
                
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
                
                if i == 0:
                    n = min(3, images.size(0))
                    samples = [
                        images[:n].cpu(),
                        masks[:n].cpu(), 
                        preds[:n].cpu(),
                        names[:n]
                    ]
        
        all_preds = torch.cat(all_preds)
        all_masks = torch.cat(all_masks)
        iou = self.calc_iou(all_preds, all_masks)
        
        if samples:
            img, true, pred, names = samples
            fig = Visualizer.create_grid(img, true, pred, len(img))
            
            tb_img = Visualizer.fig_to_tensor(fig)
            self.writer.add_image('Validation/Grid', tb_img, self.epoch)
            
            viz_path = Visualizer.save_fig(fig, self.epoch)
            print(f"Viz saved: {viz_path}")
        
        self.model.train()
        return iou

    def check_overfit(self):
        if len(self.val_iou) < 5:
            return
        
        recent = self.val_iou[-5:]
        if all(recent[i] <= recent[i-1] for i in range(1, 5)):
            best_epoch = np.argmax(self.val_iou) + 1
            best_score = max(self.val_iou)
            print(f"Overfit warning! Best: epoch {best_epoch} (IoU: {best_score:.4f})")

    def train(self):
        if not self.load_state():
            print("Starting fresh training...")
        
        print(f"Training for {Config.EPOCHS} epochs...")
        
        try:
            for epoch in range(self.epoch, Config.EPOCHS + 1):
                self.epoch = epoch
                
                # Train
                loss = self.train_epoch()
                self.train_loss.append(loss)
                
                # Validate
                iou = self.validate()
                self.val_iou.append(iou)
                
                print(f"Epoch {epoch}: Loss={loss:.4f}, IoU={iou:.4f}")
                
                # Logging
                self._update_log(epoch, loss, iou)
                self.writer.add_scalar('Loss/Train', loss, epoch)
                self.writer.add_scalar('Metrics/IoU', iou, epoch)
                
                # Save best
                if iou > self.best_iou:
                    self.best_iou = iou
                    self.save_ckpt('best_model', best=True)
                
                # Periodic checkpoint
                if epoch % Config.CKPT_FREQ == 0:
                    self.save_ckpt(f'epoch_{epoch:03d}')
                    self.check_overfit()
                
                # Save state
                self.save_state()
                
        except Exception as e:
            print(f"Error: {e}")
            self.save_state()
            raise
        
        finally:
            self.writer.close()
            print(f"Training complete! Best IoU: {self.best_iou:.4f}")

# 6. MAIN
def main():
    trainer = Trainer()
    trainer.train()

if __name__ == "__main__":
    main()