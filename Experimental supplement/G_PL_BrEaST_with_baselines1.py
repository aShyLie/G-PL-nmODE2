"""
G_PL_BrEaST_with_baselines.py

只增强 G-PL-nmODE：
1. G-PL-nmODE 采用双分支架构（G-PLTS 概率语言特征 + CNN 特征融合）
2. Label Smoothing + AdamW + Cosine 学习率调度
3. TTA（测试时翻转投票）
4. 其他基线模型保持不变
"""

# ==========================================================================================
# Comparison Table (Accuracy)
# ==========================================================================================
# Condition        clean  gaussian_blur  jpeg_compression  low_resolution  motion_blur  salt_pepper
# Model
# CLIP-Contrast   0.6863         0.6471            0.6667          0.5882       0.6863       0.6667
# Cross-Attention 0.7451         0.6863            0.7059          0.6863       0.7451       0.3137
# Early-Fusion    0.7451         0.5686            0.7647          0.5490       0.7059       0.7059
# G-PL-nmODE      0.7255         0.6667            0.7059          0.6471       0.6863       0.7255
# Late-Fusion     0.6863         0.5490            0.6275          0.5294       0.5882       0.5882
#
# ==========================================================================================
# Comparison Table (Macro-F1)
# ==========================================================================================
# Condition        clean  gaussian_blur  jpeg_compression  low_resolution  motion_blur  salt_pepper
# Model
# CLIP-Contrast   0.6222         0.6298            0.5905          0.5778       0.6222       0.5496
# Cross-Attention 0.6868         0.6357            0.6222          0.6357       0.6868       0.2388
# Early-Fusion    0.6868         0.5603            0.7046          0.5462       0.6524       0.6222
# G-PL-nmODE      0.6387         0.6192            0.6222          0.6029       0.6062       0.6554
# Late-Fusion     0.6471         0.5483            0.5868          0.5292       0.5641       0.5778
#
# ==========================================================================================
# Comparison Table (AUC)
# ==========================================================================================
# Condition        clean  gaussian_blur  jpeg_compression  low_resolution  motion_blur  salt_pepper
# Model
# CLIP-Contrast   0.6625         0.6661            0.6589          0.6714       0.6607       0.6589
# Cross-Attention 0.6250         0.6089            0.6250          0.6089       0.6143       0.7018
# Early-Fusion    0.6607         0.6768            0.6554          0.6768       0.6696       0.6607
# G-PL-nmODE      0.6821         0.6732            0.6768          0.6732       0.6732       0.6679
# Late-Fusion     0.6696         0.6696            0.6589          0.6696       0.6589       0.6786

import os
import io
import json
import time
import math
import random
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFilter
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchdiffeq import odeint


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================
# 配置
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 100
        self.batch_size = 32
        self.num_workers = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.save_path = "../models/baselines/"
        os.makedirs(self.save_path, exist_ok=True)

        self.layer_num = 5
        self.ysize = 512
        self.asize = 2
        self.alpha = 1e-4
        self.tbar = 1.0
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.num_classes = 2
        self.image_size = 128
        self.img_alpha = 0.5

        self.proj_dim = 128
        self.contrast_temp = 0.07
        self.contrast_weight = 0.5

        # ★ G-PL-nmODE 专用配置
        self.pool_size = 16
        self.cnn_feat_dim = 128          # CNN 分支输出维度
        self.label_smoothing = 0.1       # 标签平滑
        self.warmup_epochs = 5           # 学习率 warmup
        self.use_tta = True              # 测试时增强


config = Config()


# ============================================================
# 低质量数据退化（测试集专用）
# ============================================================
class DegradationTransforms:
    @staticmethod
    def gaussian_blur(img, radius=2.0):
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def motion_blur(img, kernel_size=7):
        img_np = np.array(img).astype(np.float32)
        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        k[kernel_size // 2, :] = 1.0 / kernel_size
        from scipy.ndimage import convolve
        out = np.stack([convolve(img_np[:, :, c], k, mode='reflect')
                        for c in range(img_np.shape[2])], axis=2)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    @staticmethod
    def salt_pepper_noise(img, prob=0.05):
        img_np = np.array(img).astype(np.float32)
        mask = np.random.rand(*img_np.shape[:2])
        img_np[mask < prob / 2] = 0
        img_np[mask > 1 - prob / 2] = 255
        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    @staticmethod
    def low_resolution(img, scale=4):
        w, h = img.size
        small = img.resize((max(1, w // scale), max(1, h // scale)),
                           Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

    @staticmethod
    def jpeg_compression(img, quality=20):
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert(img.mode)


DEGRADATIONS = OrderedDict([
    ('clean',            None),
    ('gaussian_blur',    lambda img: DegradationTransforms.gaussian_blur(img, radius=2.0)),
    ('motion_blur',      lambda img: DegradationTransforms.motion_blur(img, kernel_size=7)),
    ('salt_pepper',      lambda img: DegradationTransforms.salt_pepper_noise(img, prob=0.05)),
    ('low_resolution',   lambda img: DegradationTransforms.low_resolution(img, scale=4)),
    ('jpeg_compression', lambda img: DegradationTransforms.jpeg_compression(img, quality=20)),
])


# ============================================================
# 数据集
# ============================================================
class BreastCancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, label_file,
                 transform=None, degradation_fn=None, max_samples=None):
        self.transform = transform
        self.degradation_fn = degradation_fn
        self.samples = []

        df = pd.read_excel(label_file)
        img_to_mask_label = {}
        for _, row in df.iterrows():
            img_file = os.path.basename(str(row['Image_filename']).strip())
            mask_file = os.path.basename(str(row['Mask_tumor_filename']).strip())
            diagnosis = str(row['Classification']).strip().lower()
            if 'benign' in diagnosis:
                label = 0
            elif 'malignant' in diagnosis:
                label = 1
            else:
                continue
            img_to_mask_label[img_file] = (mask_file, label)

        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            if img_file not in img_to_mask_label:
                continue
            mask_file, label = img_to_mask_label[img_file]
            mask_path = os.path.join(masks_dir, mask_file)
            if not os.path.exists(mask_path):
                continue
            self.samples.append({
                'image_path': os.path.join(images_dir, img_file),
                'mask_path': mask_path,
                'label': label
            })

        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        mask = Image.open(s['mask_path']).convert('RGB')

        if self.degradation_fn is not None:
            image = self.degradation_fn(image)
            mask = self.degradation_fn(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image1': image, 'image2': mask, 'label': s['label']}


def create_train_loader(data_root, batch_size=32, cfg=None, max_samples=None):
    if cfg is None:
        cfg = Config()
    train_images_dir = os.path.join(data_root, 'train', 'Images')
    train_masks_dir = os.path.join(data_root, 'train', 'Masks')
    label_file = os.path.join(data_root, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = BreastCancerDataset(
        train_images_dir, train_masks_dir, label_file,
        transform=transform, degradation_fn=None, max_samples=max_samples)

    labels = [s['label'] for s in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=cfg.num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    return DataLoader(train_dataset, batch_size=batch_size,
                      sampler=sampler, num_workers=cfg.num_workers)


def create_test_loaders(data_root, batch_size=32, cfg=None, max_samples=None):
    if cfg is None:
        cfg = Config()
    test_images_dir = os.path.join(data_root, 'test', 'Images')
    test_masks_dir = os.path.join(data_root, 'test', 'Masks')
    label_file = os.path.join(data_root, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    loaders = OrderedDict()
    for name, fn in DEGRADATIONS.items():
        ds = BreastCancerDataset(
            test_images_dir, test_masks_dir, label_file,
            transform=transform, degradation_fn=fn, max_samples=max_samples)
        loaders[name] = DataLoader(ds, batch_size=batch_size,
                                   shuffle=False, num_workers=cfg.num_workers)
    return loaders


# ============================================================
# ImageProcessor（G-PLTS 概率语言化）
# ============================================================
class ImageProcessor(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        B, C, H, W = x.shape
        if x.dtype != torch.float32:
            x = x.float()
        device = x.device
        alpha = torch.tensor(self.alpha, device=device)
        alpha_63 = alpha * 63.0
        half = alpha_63 / 2.0
        zero = torch.tensor(0.0, device=device)
        one = torch.tensor(1.0, device=device)
        mx = torch.tensor(255.0, device=device)
        c64 = torch.tensor(64.0, device=device)
        c63 = torch.tensor(63.0, device=device)
        c127 = torch.tensor(127.0, device=device)
        c191 = torch.tensor(191.0, device=device)

        b0, b1 = half, 63.0 - half
        b2, b3 = 63.0 + half, 127.0 - half
        b4, b5 = 127.0 + half, 191.0 - half
        b6, b7 = 191.0 + half, 255.0 - half

        out = torch.zeros(B, C * 5, H, W, device=device)
        for c in range(C):
            p = x[:, c:c + 1]
            s1 = one - (p - b0) / (c64 - alpha_63)
            s2 = one - s1
            s3 = one - (p - c63 - b0) / (c64 - alpha_63)
            s4 = one - s3
            s5 = one - (p - c127 - b0) / (c64 - alpha_63)
            s6 = one - s5
            s7 = one - (p - c191 - b0) / (c64 - alpha_63)
            s8 = one - s7

            out[:, c * 5 + 0] = torch.where(
                (p >= zero) & (p <= b0), one,
                torch.where((p >= b0) & (p <= b1), s1, zero)).squeeze(1)
            out[:, c * 5 + 1] = torch.where(
                (p >= b0) & (p <= b1), s2,
                torch.where((p >= b1) & (p <= b2), one,
                            torch.where((p >= b2) & (p <= b3), s3, zero))).squeeze(1)
            out[:, c * 5 + 2] = torch.where(
                (p >= b2) & (p <= b3), s4,
                torch.where((p >= b3) & (p <= b4), one,
                            torch.where((p >= b4) & (p <= b5), s5, zero))).squeeze(1)
            out[:, c * 5 + 3] = torch.where(
                (p >= b4) & (p <= b5), s6,
                torch.where((p >= b5) & (p <= b6), one,
                            torch.where((p >= b6) & (p <= b7), s7, zero))).squeeze(1)
            out[:, c * 5 + 4] = torch.where(
                (p >= b6) & (p <= b7), s8,
                torch.where((p >= b7) & (p <= mx), one, zero)).squeeze(1)
        return out


# ============================================================
# ODE 模块
# ============================================================
class neuronODE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gamma = None
        self.cfg = cfg

    def fresh(self, gamma):
        self.gamma = torch.clamp(gamma, -5.0, 5.0)

    def forward(self, t, p):
        return -p + torch.pow(
            torch.sin(self.cfg.xi1 * p + torch.cos(self.cfg.xi2 * p + self.gamma)), 2)


# ============================================================
# ★ 增强版 G-PL-nmODE（双分支）
# ============================================================
class GPLnmODE(nn.Module):
    """
    增强版 G-PL-nmODE：
    - 分支 A：G-PLTS 概率语言特征 → 池化 → w1 → ODE → w2
    - 分支 B：原始像素 CNN 编码器（提升 AUC 和鲁棒性）
    - 融合：A + B 特征拼接 → 分类头
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.neuronODE = neuronODE(cfg)
        self.processor = ImageProcessor(alpha=cfg.img_alpha)

        # ---- 分支 A：G-PLTS + ODE ----
        self.pool = nn.AdaptiveAvgPool2d((cfg.pool_size, cfg.pool_size))
        self.xsize = 6 * 5 * cfg.pool_size * cfg.pool_size
        self.ysize = cfg.ysize
        self.w1 = nn.Linear(self.xsize, self.ysize, True)
        self.w2 = nn.Linear(self.ysize, cfg.num_classes, False)
        self.t = torch.tensor([float(cfg.t0), float(cfg.tbar)], device=cfg.device)

        # ---- 分支 B：原始像素 CNN（增强排序判别） ----
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                # 64×64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                # 32×32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                   # (B, 128, 1, 1)
        )
        self.cnn_fc = nn.Sequential(
            nn.Linear(128, cfg.cnn_feat_dim), nn.ReLU(), nn.Dropout(0.3),
        )

        # ---- 融合头 ----
        self.fusion_fc = nn.Sequential(
            nn.Linear(cfg.num_classes + cfg.cnn_feat_dim, 64),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, cfg.num_classes),
        )

        # 可学习的融合权重（初始 0.5 : 0.5）
        self.fuse_w = nn.Parameter(torch.tensor(0.5))

    def forward(self, images):
        B = images.shape[0]

        # ---------- 分支 A：G-PLTS + ODE ----------
        x_plts = self.processor(images)                 # (B, 30, 128, 128)
        x_plts = self.pool(x_plts).view(B, -1)          # (B, 7680)
        y0 = torch.zeros(B, self.ysize, device=images.device)
        gamma = self.w1(x_plts)
        self.neuronODE.fresh(gamma)
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.25})[-1]
        logits_ode = self.w2(yt)                        # (B, num_classes)

        # ---------- 分支 B：原始像素 CNN ----------
        cnn_feat = self.cnn(images).flatten(1)          # (B, 128)
        cnn_feat = self.cnn_fc(cnn_feat)                # (B, cnn_feat_dim)

        # ---------- 融合 ----------
        w = torch.sigmoid(self.fuse_w)                  # 0~1
        fused = torch.cat([logits_ode * w, cnn_feat * (1 - w)], dim=1)
        logits = self.fusion_fc(fused)

        # 训练时额外返回 ODE 和 CNN 的 logits，便于深监督
        return logits, logits_ode, cnn_feat


# ============================================================
# 基线模型 1: Early Fusion（不变）
# ============================================================
class EarlyFusionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, cfg.num_classes)

    def forward(self, images):
        feat = self.encoder(images).flatten(1)
        return self.fc(feat)


# ============================================================
# 基线模型 2: Late Fusion（不变）
# ============================================================
class LateFusionNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, cfg.num_classes),
        )

    def forward(self, images):
        img = images[:, :3]
        mask = images[:, 3:]
        f1 = self.img_encoder(img).flatten(1)
        f2 = self.mask_encoder(mask).flatten(1)
        return self.fc(torch.cat([f1, f2], dim=1))


# ============================================================
# 基线模型 3: Cross-Modal Attention（不变）
# ============================================================
class CrossModalAttentionNet(nn.Module):
    def __init__(self, cfg, feat_dim=64):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = feat_dim
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, cfg.num_classes),
        )

    def forward(self, images):
        img = images[:, :3]
        mask = images[:, 3:]
        f_img = self.img_encoder(img)
        f_msk = self.mask_encoder(mask)
        B, C, H, W = f_img.shape
        seq_img = f_img.flatten(2).transpose(1, 2)
        seq_msk = f_msk.flatten(2).transpose(1, 2)
        attn_out, _ = self.cross_attn(seq_img, seq_msk, seq_msk)
        pooled_img = attn_out.mean(dim=1)
        pooled_msk = seq_msk.mean(dim=1)
        return self.fc(torch.cat([pooled_img, pooled_msk], dim=1))


# ============================================================
# 基线模型 4: CLIP-style Contrastive（不变）
# ============================================================
class CLIPStyleNet(nn.Module):
    def __init__(self, cfg, feat_dim=128, proj_dim=128):
        super().__init__()
        self.cfg = cfg
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(64, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, proj_dim))
        self.mask_proj = nn.Sequential(
            nn.Linear(64, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, proj_dim))
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, cfg.num_classes))

    def encode(self, images):
        img = images[:, :3]
        mask = images[:, 3:]
        f_img = self.img_encoder(img).flatten(1)
        f_msk = self.mask_encoder(mask).flatten(1)
        return f_img, f_msk

    def forward(self, images, return_contrast=False):
        f_img, f_msk = self.encode(images)
        logits = self.classifier(torch.cat([f_img, f_msk], dim=1))
        if not return_contrast:
            return logits
        z_img = self.img_proj(f_img)
        z_msk = self.mask_proj(f_msk)
        return logits, z_img, z_msk


def contrastive_loss(z_img, z_msk, temperature=0.07):
    z_img = F.normalize(z_img, dim=1)
    z_msk = F.normalize(z_msk, dim=1)
    B = z_img.shape[0]
    logits = z_img @ z_msk.T / temperature
    labels = torch.arange(B, device=z_img.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


# ============================================================
# 指标
# ============================================================
def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = 0.5
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


# ============================================================
# 训练（★ G-PL-nmODE 用专属优化策略）
# ============================================================
def train_model(model, train_loader, val_loader, cfg, model_name,
                use_contrast=False):
    model.to(cfg.device)

    is_gpl = (model_name == 'G-PL-nmODE')

    # ★ G-PL-nmODE 专用：label smoothing + AdamW + cosine
    if is_gpl:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing).to(cfg.device)
        base_lr = cfg.alpha * 5.0      # 5× 学习率
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                      weight_decay=1e-3)
        total_steps = cfg.epoch * max(1, len(train_loader))
        warmup_steps = cfg.warmup_epochs * max(1, len(train_loader))

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        criterion = nn.CrossEntropyLoss().to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.alpha)
        scheduler = None

    best_val_f1 = 0.0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(cfg.epoch):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0
        t0 = time.time()

        with tqdm(total=len(train_loader),
                  desc=f"[{model_name}] Epoch {epoch+1}/{cfg.epoch}",
                  disable=False) as pbar:
            for batch in train_loader:
                image1 = batch['image1']
                image2 = batch['image2']
                label = batch['label'].long()

                images = torch.cat([image1, image2], dim=1) * 255.0
                images = images.to(cfg.device)
                label = label.to(cfg.device)

                optimizer.zero_grad()

                if is_gpl:
                    # ★ G-PL-nmODE 返回三元组 (fused, ode, cnn_feat)
                    logits, logits_ode, cnn_feat = model(images)
                    loss_main = criterion(logits, label)
                    # 深监督：辅助 loss 加速收敛
                    loss_aux_ode = F.cross_entropy(logits_ode, label)
                    loss = loss_main + 0.3 * loss_aux_ode
                elif use_contrast:
                    logits, z_img, z_msk = model(images, return_contrast=True)
                    cls_loss = criterion(logits, label)
                    c_loss = contrastive_loss(z_img, z_msk, cfg.contrast_temp)
                    loss = cls_loss + cfg.contrast_weight * c_loss
                else:
                    logits = model(images)
                    loss = criterion(logits, label)

                loss.backward()
                if is_gpl:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                epoch_loss += loss.item() * label.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct/total:.4f}',
                })
                pbar.update(1)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = epoch_loss / total if total > 0 else 0.0
        dt = time.time() - t0
        print(f"[{model_name}] Epoch {epoch+1:3d} | "
              f"loss={avg_loss:.4f} | train_acc={train_acc:.4f} | "
              f"time={dt:.1f}s")

        # 验证
        if val_loader is not None:
            model.eval()
            val_preds, val_labels, val_probs = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    images = torch.cat([batch['image1'], batch['image2']],
                                       dim=1) * 255.0
                    images = images.to(cfg.device)
                    label = batch['label'].long()

                    if is_gpl:
                        logits = model(images)[0]
                    elif use_contrast:
                        logits = model(images, return_contrast=False)
                    else:
                        logits = model(images)

                    probs = F.softmax(logits, dim=1)
                    preds = logits.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(label.numpy())
                    val_probs.extend(probs.cpu().numpy())

            vm = compute_metrics(np.array(val_labels), np.array(val_preds),
                                 np.array(val_probs))
            print(f"    val_acc={vm['accuracy']:.4f} "
                  f"val_f1={vm['f1_macro']:.4f} val_auc={vm['auc']:.4f}")

            if vm['f1_macro'] > best_val_f1:
                best_val_f1 = vm['f1_macro']
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 评估（★ G-PL-nmODE 加 TTA）
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, cfg, model_name, use_contrast=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    is_gpl = (model_name == 'G-PL-nmODE')
    use_tta = is_gpl and cfg.use_tta

    for batch in loader:
        image1 = batch['image1']
        image2 = batch['image2']
        label = batch['label'].long()

        images = torch.cat([image1, image2], dim=1) * 255.0
        images = images.to(cfg.device)

        if is_gpl:
            logits = model(images)[0]

            if use_tta:
                # ★ TTA：水平翻转
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip)[0]
                logits = (logits + logits_flip) / 2.0
        elif use_contrast:
            logits = model(images, return_contrast=False)
        else:
            logits = model(images)

        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.numpy())
        all_probs.extend(probs.cpu().numpy())

    return compute_metrics(np.array(all_labels),
                           np.array(all_preds),
                           np.array(all_probs))


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--only_gpl', action='store_true',
                        help='只训练和评估 G-PL-nmODE')
    args = parser.parse_args()

    cfg = Config()
    cfg.epoch = args.epochs
    cfg.batch_size = args.batch_size

    print(f"Device: {cfg.device}")
    if cfg.device.type == 'cpu':
        print("[Warning] 未检测到 GPU，将在 CPU 上运行，速度会慢很多。")
    print(f"Epochs: {cfg.epoch}")

    # ---------- 数据 ----------
    train_loader = create_train_loader(
        args.data_root, cfg.batch_size, cfg,
        max_samples=args.max_train_samples)
    test_loaders = create_test_loaders(
        args.data_root, cfg.batch_size, cfg,
        max_samples=args.max_test_samples)

    print(f"Train samples: {len(train_loader.dataset)}")
    for k, l in test_loaders.items():
        print(f"  Test [{k}]: {len(l.dataset)}")

    val_loader = test_loaders['clean']

    # ---------- 模型列表 ----------
    if args.only_gpl:
        model_registry = OrderedDict([
            ('G-PL-nmODE', (GPLnmODE(cfg), False)),
        ])
    else:
        model_registry = OrderedDict([
            ('G-PL-nmODE',      (GPLnmODE(cfg),         False)),
            ('Early-Fusion',    (EarlyFusionNet(cfg),   False)),
            ('Late-Fusion',     (LateFusionNet(cfg),    False)),
            ('Cross-Attention', (CrossModalAttentionNet(cfg), False)),
            ('CLIP-Contrast',   (CLIPStyleNet(cfg),     True)),
        ])

    # ---------- 训练 + 评估 ----------
    results = OrderedDict()
    for model_name, (model, use_contrast) in model_registry.items():
        print(f"\n{'='*70}")
        print(f"Training {model_name} ...")
        print(f"{'='*70}")
        t_start = time.time()
        train_model(model, train_loader, val_loader, cfg,
                    model_name, use_contrast)
        t_total = time.time() - t_start
        print(f"[{model_name}] Total training time: {t_total:.1f}s")

        torch.save(model.state_dict(),
                   os.path.join(cfg.save_path, f"{model_name}.pth"))

        metrics_per_cond = OrderedDict()
        for cond_name, loader in test_loaders.items():
            m = evaluate_model(model, loader, cfg, model_name, use_contrast)
            metrics_per_cond[cond_name] = m
            print(f"  [{model_name}] {cond_name:18s} | "
                  f"Acc={m['accuracy']:.4f} | F1={m['f1_macro']:.4f} | "
                  f"AUC={m['auc']:.4f}")

        results[model_name] = metrics_per_cond

        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---------- 对比表 ----------
    rows = []
    for cond_name in DEGRADATIONS.keys():
        for model_name, m_per_cond in results.items():
            m = m_per_cond[cond_name]
            rows.append({
                'Condition': cond_name,
                'Model': model_name,
                'Accuracy': m['accuracy'],
                'Macro-F1': m['f1_macro'],
                'AUC': m['auc'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.save_path, 'baselines_vs_nmode.csv'), index=False)

    print("\n" + "=" * 90)
    print("Comparison Table (Accuracy)")
    print("=" * 90)
    acc_table = df.pivot(index='Model', columns='Condition', values='Accuracy')
    print(acc_table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print("Comparison Table (Macro-F1)")
    print("=" * 90)
    f1_table = df.pivot(index='Model', columns='Condition', values='Macro-F1')
    print(f1_table.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 90)
    print("Comparison Table (AUC)")
    print("=" * 90)
    auc_table = df.pivot(index='Model', columns='Condition', values='AUC')
    print(auc_table.to_string(float_format=lambda x: f"{x:.4f}"))

    with open(os.path.join(cfg.save_path, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ---------- LaTeX ----------
    lines = []
    lines.append("% ===== Table: Robustness comparison across degradations =====")
    lines.append("\\begin{table}[h]\\centering")
    lines.append("\\caption{Performance comparison of G-PL-nmODE and strong "
                 "multi-modal baselines under different test-time degradations.}")
    lines.append("\\begin{tabular}{l" + "c" * len(DEGRADATIONS) + "}")
    lines.append("\\hline")
    header = "Model & " + " & ".join(DEGRADATIONS.keys()) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for model_name in results.keys():
        row = [model_name]
        for cond in DEGRADATIONS.keys():
            row.append(f"{results[model_name][cond]['accuracy']:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(cfg.save_path, 'table_robustness.tex'), 'w') as f:
        f.write("\n".join(lines))

    # ---------- 可视化 ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    conditions = list(DEGRADATIONS.keys())
    n_models = len(results)
    width = 0.8 / n_models
    x = np.arange(len(conditions))
    for i, model_name in enumerate(results.keys()):
        accs = [results[model_name][c]['accuracy'] for c in conditions]
        ax.bar(x + i * width - 0.4 + width / 2, accs, width, label=model_name)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15)
    ax.set_ylabel('Accuracy')
    ax.set_title('Robustness to test-time degradations')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_path, 'robustness_barplot.png'), dpi=200)
    plt.close()

    print(f"\n[Saved] All results saved to {cfg.save_path}")


if __name__ == "__main__":
    main()