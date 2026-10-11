"""
ablation_study.py

消融实验：
- baseline         : 完整模型（G-PLTS + ODE + 对比学习 + 图像 + mask）
- no_ode           : 去掉 ODE，用 MLP
- no_processor     : 去掉 G-PL 概率语言变换
- no_gplts         : 去掉 G-PL 概率语言变换，用归一化代替
- no_contrast      : 去掉对比学习分支
- no_mask          : 去掉 mask 模态
- no_image         : 去掉 image 模态
- smaller_y        : ysize 减半
- shorter_ode      : tbar 缩短

修复：
- 打开 tqdm + 每 epoch 打印
- nmODE2 加入 AdaptiveAvgPool，xsize 491520 → 7680（60× 加速）
- 支持 --max_samples 快速验证
"""

# ====================================================================================================
# Paired Wilcoxon test vs baseline (Holm-Bonferroni corrected)
# ====================================================================================================
#      variant    metric  ours_mean  variant_mean  mean_diff    p_raw  p_holm  significant  cliffs_delta
#       no_ode  accuracy   0.596000      0.644000  -0.048000 0.273322     1.0        False         -0.52
#       no_ode  f1_macro   0.573848      0.633223  -0.059375 0.312500     1.0        False         -0.52
#       no_ode f1_binary   0.527097      0.579660  -0.052563 0.312500     1.0        False         -0.12
#       no_ode       auc   0.764763      0.742897   0.021866 0.625000     1.0        False          0.28
#       no_ode       fnr   0.354649      0.337895   0.016754 0.812500     1.0        False         -0.12
# no_processor  accuracy   0.596000      0.584000   0.012000 0.812500     1.0        False          0.04
# no_processor  f1_macro   0.573848      0.434538   0.139310 0.062500     1.0        False          1.00
# no_processor f1_binary   0.527097      0.144923   0.382174 0.062500     1.0        False          1.00
# no_processor       auc   0.764763      0.466724   0.298039 0.062500     1.0        False          1.00
# no_processor       fnr   0.354649      0.900088  -0.545439 0.062500     1.0        False         -1.00
#     no_gplts  accuracy   0.596000      0.584000   0.012000 0.812500     1.0        False          0.04
#     no_gplts  f1_macro   0.573848      0.434538   0.139310 0.062500     1.0        False          1.00
#     no_gplts f1_binary   0.527097      0.144923   0.382174 0.062500     1.0        False          1.00
#     no_gplts       auc   0.764763      0.466724   0.298039 0.062500     1.0        False          1.00
#     no_gplts       fnr   0.354649      0.900088  -0.545439 0.062500     1.0        False         -1.00
#  no_contrast  accuracy   0.596000      0.656000  -0.060000 0.312500     1.0        False         -0.44
#  no_contrast  f1_macro   0.573848      0.631737  -0.057889 0.187500     1.0        False         -0.52
#  no_contrast f1_binary   0.527097      0.546894  -0.019797 0.437500     1.0        False         -0.28
#  no_contrast       auc   0.764763      0.757847   0.006916 0.625000     1.0        False          0.04
#  no_contrast       fnr   0.354649      0.454474  -0.099825 0.437500     1.0        False         -0.44
#      no_mask  accuracy   0.596000      0.592000   0.004000 1.000000     1.0        False          0.04
#      no_mask  f1_macro   0.573848      0.560934   0.012915 1.000000     1.0        False          0.04
#      no_mask f1_binary   0.527097      0.458704   0.068393 0.625000     1.0        False          0.36
#      no_mask       auc   0.764763      0.598024   0.166739 0.187500     1.0        False          0.76
#      no_mask       fnr   0.354649      0.534298  -0.179649 0.273322     1.0        False         -0.56
#     no_image  accuracy   0.596000      0.628000  -0.032000 0.063318     1.0        False         -0.40
#     no_image  f1_macro   0.573848      0.609814  -0.035966 0.125000     1.0        False         -0.44
#     no_image f1_binary   0.527097      0.567063  -0.039966 0.812500     1.0        False          0.12
#     no_image       auc   0.764763      0.674642   0.090121 0.062500     1.0        False          0.76
#     no_image       fnr   0.354649      0.357895  -0.003246 0.625000     1.0        False         -0.20
#    smaller_y  accuracy   0.596000      0.652000  -0.056000 0.273322     1.0        False         -0.44
#    smaller_y  f1_macro   0.573848      0.630789  -0.056941 0.312500     1.0        False         -0.36
#    smaller_y f1_binary   0.527097      0.545783  -0.018686 0.812500     1.0        False          0.28
#    smaller_y       auc   0.764763      0.694083   0.070680 0.187500     1.0        False          0.52
#    smaller_y       fnr   0.354649      0.439649  -0.085000 0.437500     1.0        False         -0.36
#  shorter_ode  accuracy   0.596000      0.684000  -0.088000 0.062500     1.0        False         -0.68
#  shorter_ode  f1_macro   0.573848      0.656150  -0.082302 0.062500     1.0        False         -0.68
#  shorter_ode f1_binary   0.527097      0.565540  -0.038443 0.625000     1.0        False         -0.16
#  shorter_ode       auc   0.764763      0.764605   0.000158 1.000000     1.0        False          0.12
#  shorter_ode       fnr   0.354649      0.453509  -0.098860 0.465209     1.0        False         -0.40

import os
import re
import json
import copy
import time
import random
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support,
)
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torchdiffeq import odeint


# ============================================================
# 配置
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 60
        self.batch_size = 32
        self.num_workers = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.save_path = "./models/ablation/"
        os.makedirs(self.save_path, exist_ok=True)

        # 模型结构
        self.ysize = 256
        self.asize = 2
        self.image_size = 128
        self.img_alpha = 0.5
        self.dropout = 0.3

        # ★ G-PL-nmODE 池化尺寸
        self.pool_size = 16

        # ODE
        self.tbar = 0.5
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49

        # 优化
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.grad_clip = 1.0

        # 训练策略
        self.val_ratio = 0.2
        self.patience = 10
        self.scheduler_patience = 5

        # 类别权重
        self.class_weight = [1.0, 1.0]

        # 对比学习
        self.proj_dim = 64
        self.contrast_temp = 0.07
        self.contrast_weight = 0.5

        # 统计
        self.seeds = list(range(42, 52))
        self.n_bootstrap = 1000

        # 消融变体
        self.variants = [
            'baseline',
            'no_ode',
            'no_processor',
            'no_gplts',
            'no_contrast',
            'no_mask',
            'no_image',
            'smaller_y',
            'shorter_ode',
        ]


config = Config()


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


# ============================================================
# 患者级划分
# ============================================================
def extract_patient_id(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r'([A-Za-z]*\d+)', base)
    if m:
        return m.group(1)
    return base.split('_')[0]


# ============================================================
# 数据集
# ============================================================
class MultimodalBrEaSTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, label_file, transform=None,
                 use_image=True, use_mask=True, samples=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.use_image = use_image
        self.use_mask = use_mask

        if samples is not None:
            self.samples = samples
            return

        df = pd.read_excel(label_file)
        img_info = {}
        for _, row in df.iterrows():
            img_file = os.path.basename(str(row['Image_filename']).strip())
            diagnosis = str(row['Classification']).strip().lower()
            if 'benign' in diagnosis:
                img_info[img_file] = 0
            elif 'malignant' in diagnosis:
                img_info[img_file] = 1

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        mask_files = set(os.listdir(masks_dir))

        def mask_candidates(img_filename):
            stem, ext = os.path.splitext(img_filename)
            return [img_filename, f"{stem}_mask{ext}", f"{stem}_tumor{ext}",
                    f"{stem}_seg{ext}", f"mask_{img_filename}"]

        self.samples = []
        for img_file in image_files:
            key = img_file
            if key not in img_info:
                stem = os.path.splitext(key)[0]
                if stem in img_info:
                    key = stem
                else:
                    continue
            label = img_info[key]

            mask_path = None
            for cand in mask_candidates(img_file):
                p = os.path.join(masks_dir, cand)
                if p in mask_files or os.path.exists(p):
                    mask_path = p
                    break
            if mask_path is None:
                continue

            self.samples.append({
                'image_path': os.path.join(images_dir, img_file),
                'mask_path': mask_path,
                'label': label,
                'patient_id': extract_patient_id(img_file),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        mask = Image.open(s['mask_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.use_image and self.use_mask:
            return image, mask, s['label']
        elif self.use_image:
            return image, s['label']
        elif self.use_mask:
            return mask, s['label']
        else:
            raise ValueError("至少需要图像或 mask 之一")


def build_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])


def load_all_samples(data_root, max_samples=None):
    label_file = os.path.join(
        data_root, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
    all_samples = []
    for split in ['train', 'test']:
        images_dir = os.path.join(data_root, split, 'Images')
        masks_dir = os.path.join(data_root, split, 'Masks')
        if not os.path.isdir(images_dir):
            continue
        ds = MultimodalBrEaSTDataset(images_dir, masks_dir, label_file,
                                     transform=None,
                                     use_image=True, use_mask=True)
        all_samples.extend(ds.samples)

    if max_samples is not None and max_samples < len(all_samples):
        all_samples = all_samples[:max_samples]
    return all_samples


def patient_level_split(samples, seed, val_ratio=0.2, test_ratio=0.2):
    rng = random.Random(seed)
    by_patient = defaultdict(list)
    for s in samples:
        by_patient[s['patient_id']].append(s)

    patient_ids = list(by_patient.keys())
    rng.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))

    test_patients = set(patient_ids[:n_test])
    val_patients = set(patient_ids[n_test:n_test + n_val])

    train_samples, val_samples, test_samples = [], [], []
    for pid, group in by_patient.items():
        if pid in test_patients:
            test_samples.extend(group)
        elif pid in val_patients:
            val_samples.extend(group)
        else:
            train_samples.extend(group)
    return train_samples, val_samples, test_samples


def make_loader(samples, use_image, use_mask, train=False,
                batch_size=32, balanced=True):
    ds = MultimodalBrEaSTDataset(
        images_dir='', masks_dir='', label_file='',
        transform=build_transform(train),
        use_image=use_image, use_mask=use_mask,
        samples=samples)

    if balanced and train:
        labels = [s['label'] for s in samples]
        class_counts = np.bincount(labels, minlength=config.asize)
        sample_weights = [1.0 / max(class_counts[l], 1) for l in labels]
        sampler = WeightedRandomSampler(sample_weights,
                                        num_samples=len(sample_weights),
                                        replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=config.num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=config.num_workers)


# ============================================================
# ImageProcessor
# ============================================================
class ImageProcessor(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        B, C, H, W = x.shape
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
            out[:, c * 5 + 0] = torch.where((p >= zero) & (p <= b0), one,
                                            torch.where((p >= b0) & (p <= b1), s1, zero)).squeeze(1)
            out[:, c * 5 + 1] = torch.where((p >= b0) & (p <= b1), s2,
                                            torch.where((p >= b1) & (p <= b2), one,
                                                        torch.where((p >= b2) & (p <= b3), s3, zero))).squeeze(1)
            out[:, c * 5 + 2] = torch.where((p >= b2) & (p <= b3), s4,
                                            torch.where((p >= b3) & (p <= b4), one,
                                                        torch.where((p >= b4) & (p <= b5), s5, zero))).squeeze(1)
            out[:, c * 5 + 3] = torch.where((p >= b4) & (p <= b5), s6,
                                            torch.where((p >= b5) & (p <= b6), one,
                                                        torch.where((p >= b6) & (p <= b7), s7, zero))).squeeze(1)
            out[:, c * 5 + 4] = torch.where((p >= b6) & (p <= b7), s8,
                                            torch.where((p >= b7) & (p <= mx), one, zero)).squeeze(1)
        return out


# ============================================================
# ODE 模块
# ============================================================
class neuronODE(nn.Module):
    def __init__(self, xi1=0.49, xi2=0.49):
        super().__init__()
        self.xi1 = xi1
        self.xi2 = xi2
        self.gamma = None

    def fresh(self, gamma):
        self.gamma = torch.clamp(gamma, -5.0, 5.0)

    def forward(self, t, p):
        return -p + torch.pow(
            torch.sin(self.xi1 * p + torch.cos(self.xi2 * p + self.gamma)), 2)


# ============================================================
# 主模型（★ 加池化 + 对比分支独立 FC）
# ============================================================
class nmODE2(nn.Module):
    def __init__(self, input_channels, ysize=256, asize=2, dropout=0.3,
                 t0=0, tbar=0.5, xi1=0.49, xi2=0.49,
                 use_contrast=True, use_ode=True, use_processor=True,
                 n_modality=2, pool_size=16):
        super().__init__()
        self.use_contrast = use_contrast
        self.use_ode = use_ode
        self.use_processor = use_processor
        self.n_modality = n_modality
        self.pool_size = pool_size

        self.input_channels = input_channels
        self.img_size = config.image_size

        # ★ 池化：把 H×W 降到 pool_size×pool_size
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.xsize = input_channels * pool_size * pool_size
        self.ysize = ysize
        self.asize = asize

        # G-PL 变换
        self.processor = (ImageProcessor(alpha=config.img_alpha)
                          if use_processor else nn.Identity())

        # 特征映射（主分支）
        self.w1 = nn.Linear(self.xsize, self.ysize, True)
        self.w2 = nn.Linear(self.ysize, self.asize, False)
        self.dropout = nn.Dropout(dropout)

        # ODE 模块
        if use_ode:
            self.neuronODE = neuronODE(xi1, xi2)
            self.t = torch.tensor([float(t0), float(tbar)],
                                  device=config.device)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.xsize, self.ysize),
                nn.Dropout(dropout), nn.ReLU(),
                nn.Linear(self.ysize, self.asize))

        # 对比学习分支：为单模态单独建立映射
        if use_contrast:
            single_ch = max(1, input_channels // max(n_modality, 1))
            self.single_pool = nn.AdaptiveAvgPool2d(1)
            self.single_fc = nn.Linear(single_ch, config.ysize)
            self.img_proj = nn.Sequential(
                nn.Linear(config.ysize, config.proj_dim), nn.ReLU(),
                nn.Linear(config.proj_dim, config.proj_dim))
            self.mask_proj = nn.Sequential(
                nn.Linear(config.ysize, config.proj_dim), nn.ReLU(),
                nn.Linear(config.proj_dim, config.proj_dim))

    def forward(self, image, mask=None, return_contrast=False):
        B = image.shape[0]

        img_p = self.processor(image) if self.use_processor else image
        if mask is not None:
            msk_p = self.processor(mask) if self.use_processor else mask
            x = torch.cat([img_p, msk_p], dim=1)
        else:
            msk_p = None
            x = img_p

        # ★ 池化后展平
        x = self.pool(x).view(B, -1)

        # ---- 无 ODE ----
        if not self.use_ode:
            logits = self.mlp(x)
            if not return_contrast or not self.use_contrast:
                return logits
            f_img = self.single_pool(img_p).flatten(1)
            z_img = self.img_proj(self.single_fc(f_img))
            if msk_p is not None:
                f_msk = self.single_pool(msk_p).flatten(1)
                z_msk = self.mask_proj(self.single_fc(f_msk))
            else:
                z_msk = z_img
            return logits, z_img, z_msk

        # ---- 有 ODE ----
        gamma = self.w1(x)
        gamma = self.dropout(gamma)
        self.neuronODE.fresh(gamma)
        y0 = torch.zeros(B, self.ysize, device=x.device)
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.1})[-1]
        logits = self.w2(yt)

        if not return_contrast or not self.use_contrast:
            return logits

        f_img = self.single_pool(img_p).flatten(1)
        z_img = self.img_proj(self.single_fc(f_img))
        if msk_p is not None:
            f_msk = self.single_pool(msk_p).flatten(1)
            z_msk = self.mask_proj(self.single_fc(f_msk))
        else:
            z_msk = z_img
        return logits, z_img, z_msk


# ============================================================
# InfoNCE 对比损失
# ============================================================
def info_nce(z_a, z_b, temperature=0.07):
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    B = z_a.shape[0]
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(B, device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))


# ============================================================
# 变体 → flags
# ============================================================
def variant_to_flags(variant):
    cfg = {
        'use_image': True, 'use_mask': True,
        'use_processor': True, 'use_ode': True, 'use_contrast': True,
        'ysize': config.ysize, 'tbar': config.tbar,
    }
    if variant == 'baseline':
        pass
    elif variant == 'no_ode':
        cfg['use_ode'] = False
    elif variant == 'no_processor':
        cfg['use_processor'] = False
    elif variant == 'no_gplts':
        cfg['use_processor'] = False
    elif variant == 'no_contrast':
        cfg['use_contrast'] = False
    elif variant == 'no_mask':
        cfg['use_mask'] = False
    elif variant == 'no_image':
        cfg['use_image'] = False
    elif variant == 'smaller_y':
        cfg['ysize'] = config.ysize // 2
    elif variant == 'shorter_ode':
        cfg['tbar'] = config.tbar / 5.0
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        if len(batch) == 3:
            image, mask, label = batch
            image = image.to(device).float() * 255.0
            mask = mask.to(device).float() * 255.0
        else:
            image, label = batch
            image = image.to(device).float() * 255.0
            mask = None

        out = model(image, mask, return_contrast=False)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(label.numpy())
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    _, _, f1_bin, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0

    return {
        'accuracy': float(acc),
        'f1_macro': float(f1m),
        'f1_binary': float(f1_bin),
        'auc': float(auc),
        'fnr': float(fnr),
    }


# ============================================================
# 单次实验（★ 加 tqdm + 每 epoch 打印）
# ============================================================
def run_one_seed(variant, seed, samples, args):
    set_seed(seed)
    flags = variant_to_flags(variant)
    device = config.device

    train_samples, val_samples, test_samples = patient_level_split(
        samples, seed=seed, val_ratio=config.val_ratio, test_ratio=0.2)
    assert set(s['patient_id'] for s in train_samples).isdisjoint(
        set(s['patient_id'] for s in test_samples)), "患者泄漏"

    train_loader = make_loader(train_samples, flags['use_image'], flags['use_mask'],
                               train=True, batch_size=args.batch_size)
    val_loader = make_loader(val_samples, flags['use_image'], flags['use_mask'],
                             train=False, batch_size=args.batch_size)
    test_loader = make_loader(test_samples, flags['use_image'], flags['use_mask'],
                              train=False, batch_size=args.batch_size)

    print(f"[{variant}/{seed}] train={len(train_samples)} "
          f"val={len(val_samples)} test={len(test_samples)}")

    # 输入通道数
    if flags['use_image'] and flags['use_mask']:
        raw_channels = 6
        n_modality = 2
    else:
        raw_channels = 3
        n_modality = 1
    input_channels = raw_channels * (5 if flags['use_processor'] else 1)

    model = nmODE2(
        input_channels=input_channels,
        ysize=flags['ysize'], asize=config.asize, dropout=config.dropout,
        t0=config.t0, tbar=flags['tbar'],
        xi1=config.xi1, xi2=config.xi2,
        use_contrast=flags['use_contrast'],
        use_ode=flags['use_ode'],
        use_processor=flags['use_processor'],
        n_modality=n_modality,
        pool_size=config.pool_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_weight, dtype=torch.float32).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=config.scheduler_patience, factor=0.5)

    best_val_f1 = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        # ★ 打开 tqdm
        with tqdm(total=len(train_loader),
                  desc=f"[{variant}/{seed}] Epoch {epoch+1}/{args.epochs}",
                  disable=False) as pbar:
            for batch in train_loader:
                if len(batch) == 3:
                    image, mask, label = batch
                    image = image.to(device).float() * 255.0
                    mask = mask.to(device).float() * 255.0
                else:
                    image, label = batch
                    image = image.to(device).float() * 255.0
                    mask = None
                label = label.to(device)

                optimizer.zero_grad()
                if flags['use_contrast']:
                    logits, z_img, z_msk = model(image, mask, return_contrast=True)
                    cls_loss = criterion(logits, label)
                    if z_img is not None and z_msk is not None:
                        c_loss = info_nce(z_img, z_msk, config.contrast_temp)
                        loss = cls_loss + config.contrast_weight * c_loss
                    else:
                        loss = cls_loss
                else:
                    out = model(image, mask, return_contrast=False)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = criterion(out, label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)

        # 验证
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics['f1_macro'])

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        dt = time.time() - t0
        print(f"[{variant}/{seed}] Epoch {epoch+1:3d} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1m={val_metrics['f1_macro']:.4f} | "
              f"best_f1={best_val_f1:.4f} | "
              f"no_improve={no_improve} | time={dt:.1f}s")

        if no_improve >= config.patience:
            print(f"[{variant}/{seed}] Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    test_metrics['variant'] = variant
    test_metrics['seed'] = seed
    test_metrics['best_val_f1'] = float(best_val_f1)
    test_metrics['n_test'] = int(len(test_samples))
    return test_metrics


# ============================================================
# Bootstrap 95% CI
# ============================================================
def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    mean = v.mean()
    std = v.std(ddof=1) if len(v) > 1 else 0.0
    boots = np.array([v[rng.integers(0, len(v), size=len(v))].mean()
                      for _ in range(n_boot)])
    boots.sort()
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return float(mean), float(std), float(lo), float(hi)


# ============================================================
# 统计检验
# ============================================================
def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))


def paired_test(baseline_dict, variants_dict, alpha=0.05):
    all_p = []
    keys = []
    delta_values = []
    for variant, metrics in variants_dict.items():
        for metric_name, ours in baseline_dict.items():
            them = metrics[metric_name]
            try:
                _, p = wilcoxon(ours, them, alternative='two-sided')
            except ValueError:
                p = 1.0
            all_p.append(p)
            keys.append((variant, metric_name))
            delta_values.append(cliffs_delta(ours, them))

    if not all_p:
        return pd.DataFrame()

    _, p_corr, _, _ = multipletests(all_p, alpha=alpha, method='holm')

    rows = []
    for i, (variant, metric_name) in enumerate(keys):
        rows.append({
            'variant': variant,
            'metric': metric_name,
            'ours_mean': float(np.mean(baseline_dict[metric_name])),
            'variant_mean': float(np.mean(variants_dict[variant][metric_name])),
            'mean_diff': float(np.mean(baseline_dict[metric_name]) -
                               np.mean(variants_dict[variant][metric_name])),
            'p_raw': float(all_p[i]),
            'p_holm': float(p_corr[i]),
            'significant': bool(p_corr[i] < alpha),
            'cliffs_delta': float(delta_values[i]),
        })
    return pd.DataFrame(rows)


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=config.epoch)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--runs', type=int, default=len(config.seeds))
    parser.add_argument('--max_samples', type=int, default=None,
                        help='只用前 N 个样本（快速验证）')
    parser.add_argument('--variants', type=str, default=None,
                        help='逗号分隔的变体列表，默认全部')
    args = parser.parse_args()

    config.epoch = args.epochs
    config.batch_size = args.batch_size
    config.seeds = list(range(42, 42 + args.runs))

    if args.variants is not None:
        config.variants = args.variants.split(',')

    print(f"Device: {config.device}")
    if config.device.type == 'cpu':
        print("[Warning] 未检测到 GPU，将在 CPU 上运行，速度会慢很多。")
        print("          建议先快速验证：")
        print("          python ablation_study.py ^")
        print("              --data_root D:\\gap\\gap-data\\BrEaST ^")
        print("              --epochs 3 --runs 1 --max_samples 30 ^")
        print("              --variants baseline,no_ode")
    print(f"Seeds : {config.seeds}")
    print(f"Epochs: {config.epoch}")
    print(f"Variants: {config.variants}")

    print("\n[Data] Loading all samples ...")
    samples = load_all_samples(args.data_root, max_samples=args.max_samples)
    print(f"[Data] Total samples: {len(samples)}")
    print(f"[Data] Approx. patients: "
          f"{len(set(s['patient_id'] for s in samples))}")

    all_results = OrderedDict()
    for variant in config.variants:
        all_results[variant] = []
        for seed in config.seeds:
            print(f"\n{'='*70}")
            print(f"Variant={variant}  Seed={seed}")
            print(f"{'='*70}")
            try:
                r = run_one_seed(variant, seed, samples, args)
                all_results[variant].append(r)
                print(f"[{variant}/{seed}] "
                      f"Acc={r['accuracy']:.4f} "
                      f"F1m={r['f1_macro']:.4f} "
                      f"AUC={r['auc']:.4f}")
            except Exception as e:
                print(f"[{variant}/{seed}] FAILED: {e}")
                import traceback
                traceback.print_exc()

    # 汇总
    print("\n" + "=" * 100)
    print("Ablation summary (mean ± std, 95% CI)")
    print("=" * 100)

    metric_keys = ['accuracy', 'f1_macro', 'f1_binary', 'auc', 'fnr']
    summary_rows = []
    for variant, runs in all_results.items():
        if not runs:
            continue
        row = {'variant': variant}
        for m in metric_keys:
            vals = [r[m] for r in runs]
            mean, std, lo, hi = bootstrap_ci(vals, n_boot=config.n_bootstrap)
            row[f'{m}_mean'] = mean
            row[f'{m}_std'] = std
            row[f'{m}_ci_lo'] = lo
            row[f'{m}_ci_hi'] = hi
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(config.save_path, 'ablation_summary.csv'),
                      index=False)

    for row in summary_rows:
        print(f"\n[{row['variant']}]")
        for m in metric_keys:
            print(f"  {m:12s}: "
                  f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}  "
                  f"95% CI [{row[f'{m}_ci_lo']:.4f}, {row[f'{m}_ci_hi']:.4f}]")

    # 配对检验
    if 'baseline' in all_results and len(all_results['baseline']) > 0:
        baseline_metrics = {
            m: np.array([r[m] for r in all_results['baseline']])
            for m in metric_keys
        }
        variants_for_test = {
            v: {m: np.array([r[m] for r in runs]) for m in metric_keys}
            for v, runs in all_results.items()
            if v != 'baseline' and len(runs) > 0
        }
        test_df = paired_test(baseline_metrics, variants_for_test)
        test_df.to_csv(os.path.join(config.save_path, 'ablation_paired_test.csv'),
                       index=False)
        print("\n" + "=" * 100)
        print("Paired Wilcoxon test vs baseline (Holm-Bonferroni corrected)")
        print("=" * 100)
        print(test_df.to_string(index=False))

    # LaTeX
    tex_lines = []
    tex_lines.append("% ===== Ablation Table =====")
    tex_lines.append("\\begin{table}[h]\\centering")
    tex_lines.append("\\caption{Ablation study. Results are reported as "
                     "mean $\\pm$ std with 95\\% CI over "
                     + str(len(config.seeds)) + " patient-level splits.}")
    tex_lines.append("\\begin{tabular}{l" + "c" * len(metric_keys) + "}")
    tex_lines.append("\\hline")
    tex_lines.append("Variant & " + " & ".join(
        ['Acc.', 'F1m', 'F1b', 'AUC', 'FNR']) + " \\\\")
    tex_lines.append("\\hline")
    for row in summary_rows:
        cells = [row['variant']]
        for m in metric_keys:
            cells.append(f"{row[f'{m}_mean']:.3f}$\\pm${row[f'{m}_std']:.3f}")
        tex_lines.append(" & ".join(cells) + " \\\\")
    tex_lines.append("\\hline")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")

    with open(os.path.join(config.save_path, 'table_ablation.tex'),
              'w', encoding='utf-8') as f:
        f.write("\n".join(tex_lines))

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, ['accuracy', 'f1_macro', 'auc']):
        variants = [r['variant'] for r in summary_rows]
        means = [r[f'{m}_mean'] for r in summary_rows]
        stds = [r[f'{m}_std'] for r in summary_rows]
        colors = ['tab:red' if v == 'baseline' else 'tab:blue'
                  for v in variants]
        ax.barh(variants, means, xerr=stds, color=colors, alpha=0.8,
                error_kw={'capsize': 3, 'alpha': 0.6})
        ax.set_xlabel(m)
        ax.set_title(f'Ablation: {m}')
        ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_path, 'ablation_barplot.png'),
                dpi=200)
    plt.close()

    # 保存 JSON
    with open(os.path.join(config.save_path, 'ablation_all_results.json'),
              'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'seeds': config.seeds,
                'epochs': config.epoch,
                'n_bootstrap': config.n_bootstrap,
                'variants': config.variants,
            },
            'per_run': {v: runs for v, runs in all_results.items()},
            'summary': summary_rows,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] All results saved to {config.save_path}")


if __name__ == "__main__":
    main()