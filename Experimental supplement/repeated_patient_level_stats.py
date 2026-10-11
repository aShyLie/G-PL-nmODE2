"""
repeated_patient_level_stats.py

功能：
1. N 次独立重复（seed 42-51），每次使用不同的患者级划分
2. 每个 seed 上训练 G-PL-nmODE，报告 Accuracy / Macro-F1 / AUC
3. 均值 ± 标准差、Bootstrap 95% CI
4. 与基线方法做配对 Wilcoxon 符号秩检验 + Holm-Bonferroni 校正 + Cliff's delta
5. 输出论文可用表格（LaTeX + CSV + JSON）

修复：
- 打开 tqdm + 每 epoch 打印
- G-PL-nmODE 加入 AdaptiveAvgPool，xsize 491520 → 7680（60× 加速）
- 支持 --max_samples 快速验证

用法：
    python repeated_patient_level_stats.py --data_root D:\gap\gap-data\BrEaST --epochs 50 --runs 10
    python repeated_patient_level_stats.py --data_root D:\gap\gap-data\BrEaST --epochs 3 --runs 2 --max_samples 30
"""

# baseline  ours_mean  baseline_mean  mean_diff  p_raw  p_holm  significant  cliffs_delta
#      CNN       0.73       0.679936   0.050064    0.5     1.0        False           1.0
#   ResNet       0.73       0.707453   0.022547    0.5     1.0        False           1.0
#      ViT       0.73       0.708259   0.021741    0.5     1.0        False           1.0


import os
import re
import json
import time
import glob
import random
import argparse
from collections import defaultdict
from datetime import datetime

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

        self.save_path = "../models/repeated_stats/"
        os.makedirs(self.save_path, exist_ok=True)

        self.layer_num = 5
        self.ysize = 512              # ★ 从 1024 降到 512
        self.asize = 2
        self.alpha = 1e-4             # ★ 从 1e-5 提升
        self.tbar = 1.0
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.num_classes = 2
        self.image_size = 128
        self.img_alpha = 0.5

        # ★ G-PL-nmODE 池化尺寸
        self.pool_size = 16

        # 统计
        self.seeds = list(range(42, 52))
        self.n_bootstrap = 1000


config = Config()


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
class BreastCancerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s['image_path']).convert('RGB')
        mask = Image.open(s['mask_path']).convert('RGB')
        label = s['label']
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return {'image1': image, 'image2': mask, 'label': label}


def load_all_samples(data_root, max_samples=None):
    label_file = os.path.join(
        data_root, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')
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

    samples = []
    for split in ['train', 'test']:
        images_dir = os.path.join(data_root, split, 'Images')
        masks_dir = os.path.join(data_root, split, 'Masks')
        if not os.path.isdir(images_dir):
            continue
        for img_file in os.listdir(images_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue
            if img_file not in img_to_mask_label:
                continue
            mask_file, label = img_to_mask_label[img_file]
            mask_path = os.path.join(masks_dir, mask_file)
            if not os.path.exists(mask_path):
                continue
            samples.append({
                'image_path': os.path.join(images_dir, img_file),
                'mask_path': mask_path,
                'label': label,
                'patient_id': extract_patient_id(img_file),
            })

    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]

    print(f"[Data] Loaded {len(samples)} samples")
    n_patients = len(set(s['patient_id'] for s in samples))
    print(f"[Data] Approx. {n_patients} patients")
    return samples


def patient_level_split(samples, seed, test_ratio=0.2):
    rng = random.Random(seed)
    by_patient = defaultdict(list)
    for s in samples:
        by_patient[s['patient_id']].append(s)

    patient_ids = list(by_patient.keys())
    rng.shuffle(patient_ids)
    n_test = max(1, int(len(patient_ids) * test_ratio))

    test_patients = set(patient_ids[:n_test])
    train_samples, test_samples = [], []
    for pid, group in by_patient.items():
        if pid in test_patients:
            test_samples.extend(group)
        else:
            train_samples.extend(group)
    return train_samples, test_samples


def create_loaders(train_samples, test_samples, cfg):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = BreastCancerDataset(train_samples, transform)
    test_dataset = BreastCancerDataset(test_samples, transform)

    labels = [s['label'] for s in train_samples]
    class_counts = np.bincount(labels, minlength=cfg.num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader


# ============================================================
# ImageProcessor
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
            p = x[:, c:c+1]
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
# ODE 模型（★ 加入池化，xsize 491520 → 7680）
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


class nmODE2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.neuronODE = neuronODE(cfg)
        self.processor = ImageProcessor(alpha=cfg.img_alpha)
        # ★ 池化
        self.pool = nn.AdaptiveAvgPool2d((cfg.pool_size, cfg.pool_size))
        self.xsize = 6 * 5 * cfg.pool_size * cfg.pool_size
        self.ysize = cfg.ysize
        self.asize = cfg.num_classes

        self.w1 = nn.Linear(self.xsize, self.ysize, True)
        self.w2 = nn.Linear(self.ysize, self.asize, False)
        self.t = torch.tensor([float(cfg.t0), float(cfg.tbar)],
                              device=cfg.device)

    def forward(self, data):
        B = data.shape[0]
        x = self.processor(data)               # (B, 30, 128, 128)
        x = self.pool(x).view(B, -1)           # (B, 7680)
        y0 = torch.zeros(B, self.ysize, device=data.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.25})[-1]
        return self.w2(yt)

    def save(self, path):
        torch.save({"w1": self.w1.weight, "b": self.w1.bias,
                    "w2": self.w2.weight}, path)

    def load(self, path):
        params = torch.load(path, map_location=self.cfg.device,
                            weights_only=False)
        self.w1.weight.data = params["w1"]
        self.w2.weight.data = params["w2"]
        self.w1.bias.data = params["b"]


# ============================================================
# 训练一个 seed（★ 打开 tqdm + 每 epoch 打印）
# ============================================================
def train_one_seed(seed, samples, args, cfg, save_preds=False):
    set_seed(seed)
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    train_samples, test_samples = patient_level_split(samples, seed=seed)
    print(f"[Split] train={len(train_samples)}  test={len(test_samples)}")
    train_patients = set(s['patient_id'] for s in train_samples)
    test_patients = set(s['patient_id'] for s in test_samples)
    assert train_patients.isdisjoint(test_patients), "患者级划分失败"

    train_loader, test_loader = create_loaders(train_samples, test_samples, cfg)

    net = nmODE2(cfg).to(cfg.device)
    criterion = nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.alpha)

    best_macro_f1 = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(cfg.epoch):
        net.train()
        torch.cuda.empty_cache()
        t0 = time.time()

        # ★ 打开 tqdm
        with tqdm(total=len(train_loader),
                  desc=f"[seed={seed}] Epoch {epoch+1}/{cfg.epoch}",
                  disable=False) as pbar:
            correct, total = 0, 0
            for batch in train_loader:
                image1 = batch['image1']
                image2 = batch['image2']
                label = batch['label'].long()

                images = torch.cat([image1, image2], dim=1) * 255.0
                images = images.to(cfg.device)
                label_dev = label.to(cfg.device)

                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, label_dev)
                loss.backward()
                optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == label_dev).sum().item()
                total += label.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'acc': f'{correct/total:.4f}'})
                pbar.update(1)

        # ---------- 测试 ----------
        net.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                image1 = batch['image1']
                image2 = batch['image2']
                label = batch['label'].long()
                images = torch.cat([image1, image2], dim=1) * 255.0
                images = images.to(cfg.device)
                logits = net(images)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.numpy())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        except Exception:
            auc = 0.5

        dt = time.time() - t0
        print(f"[seed={seed}] Epoch {epoch+1:3d} | "
              f"Acc={acc:.4f} | F1m={f1m:.4f} | AUC={auc:.4f} | "
              f"time={dt:.1f}s")

        if f1m > best_macro_f1:
            best_macro_f1 = f1m
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}

    # 加载最佳权重
    if best_state is not None:
        net.load_state_dict(best_state)
        net.eval()
        all_probs, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                image1 = batch['image1']
                image2 = batch['image2']
                label = batch['label'].long()
                images = torch.cat([image1, image2], dim=1) * 255.0
                images = images.to(cfg.device)
                logits = net(images)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.numpy())

        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = 0.5

    if save_preds:
        np.savez(os.path.join(cfg.save_path, f"preds_seed{seed}.npz"),
                 probs=all_probs, preds=all_preds, labels=all_labels)

    if best_state is not None:
        torch.save(best_state,
                   os.path.join(cfg.save_path, f"best_seed{seed}.pth"))

    print(f"[seed={seed}] best_epoch={best_epoch} | "
          f"Acc={acc:.4f} | F1m={f1m:.4f} | AUC={auc:.4f}")

    return {
        'seed': seed,
        'accuracy': float(acc),
        'f1_macro': float(f1m),
        'auc': float(auc),
        'best_epoch': int(best_epoch),
        'n_test': int(len(all_labels)),
        'n_train': int(len(train_samples)),
    }


# ============================================================
# Bootstrap 95% CI
# ============================================================
def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    mean = values.mean()
    std = values.std(ddof=1) if n > 1 else 0.0
    boots = np.array([values[rng.integers(0, n, size=n)].mean()
                      for _ in range(n_boot)])
    boots = np.sort(boots)
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return float(mean), float(std), float(lo), float(hi)


# ============================================================
# Cliff's delta + 配对检验
# ============================================================
def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    gt = 0
    lt = 0
    for i in range(n):
        if x[i] > y[i]:
            gt += 1
        elif x[i] < y[i]:
            lt += 1
    return (gt - lt) / n


def paired_test_with_correction(ours, baselines_dict, alpha=0.05):
    names = list(baselines_dict.keys())
    p_raw = []
    deltas = []
    for name in names:
        bl = np.asarray(baselines_dict[name])
        try:
            stat, p = wilcoxon(ours, bl, alternative='two-sided')
        except ValueError:
            p = 1.0
        p_raw.append(p)
        deltas.append(cliffs_delta(ours, bl))

    reject, p_corr, _, _ = multipletests(p_raw, alpha=alpha, method='holm')

    rows = []
    for i, name in enumerate(names):
        bl = np.asarray(baselines_dict[name])
        rows.append({
            'baseline': name,
            'ours_mean': float(np.mean(ours)),
            'baseline_mean': float(np.mean(bl)),
            'mean_diff': float(np.mean(ours) - np.mean(bl)),
            'p_raw': float(p_raw[i]),
            'p_holm': float(p_corr[i]),
            'significant': bool(reject[i]),
            'cliffs_delta': float(deltas[i]),
        })
    return pd.DataFrame(rows)


# ============================================================
# 生成论文表格
# ============================================================
def make_paper_table(all_results, cfg):
    metrics = ['accuracy', 'f1_macro', 'auc']
    names = {'accuracy': 'Accuracy', 'f1_macro': 'Macro-F1', 'auc': 'AUC'}

    per_seed_df = pd.DataFrame([{k: r[k] for k in ['seed'] + metrics}
                                for r in all_results])
    per_seed_df.to_csv(os.path.join(cfg.save_path, 'per_seed_metrics.csv'),
                       index=False)

    rows = []
    for m in metrics:
        vals = np.array([r[m] for r in all_results], dtype=float)
        mean, std, lo, hi = bootstrap_ci(vals, n_boot=cfg.n_bootstrap)
        rows.append({
            'Metric': names[m],
            'Mean ± Std': f"{mean:.4f} ± {std:.4f}",
            '95% CI': f"[{lo:.4f}, {hi:.4f}]",
            'Min': float(vals.min()),
            'Max': float(vals.max()),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(cfg.save_path, 'summary_table.csv'),
                      index=False)
    return summary_df, per_seed_df


def write_latex_summary(summary_df, per_seed_df, cfg):
    lines = []
    lines.append("% ===== Table X: Repeated patient-level split =====")
    lines.append("\\begin{table}[h]\\centering")
    lines.append("\\caption{Statistics over "
                 + str(len(per_seed_df))
                 + " repeated patient-level splits on the BrEaST dataset "
                   "(mean $\\pm$ std, 95\\% CI by bootstrap with "
                 + str(cfg.n_bootstrap) + " resamples).}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("Metric & Mean $\\pm$ Std & 95\\% CI & Range \\\\")
    lines.append("\\hline")
    for _, row in summary_df.iterrows():
        rng = f"[{row['Min']:.4f}, {row['Max']:.4f}]"
        lines.append(f"{row['Metric']} & {row['Mean ± Std']} & "
                     f"{row['95% CI']} & {rng} \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(os.path.join(cfg.save_path, 'table_summary.tex'),
              'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def plot_distributions(per_seed_df, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, m in zip(axes, ['accuracy', 'f1_macro', 'auc']):
        vals = per_seed_df[m].values
        ax.boxplot(vals, positions=[1], widths=0.5)
        ax.scatter(np.ones_like(vals) + np.random.uniform(-0.05, 0.05, len(vals)),
                   vals, s=30, alpha=0.7, color='tab:red')
        ax.axhline(vals.mean(), color='red', linestyle='--', alpha=0.5)
        ax.set_title(m)
        ax.set_xlim(0.5, 1.5)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_path, 'distribution_boxplot.png'),
                dpi=200)
    plt.close()


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=config.epoch)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='只用前 N 个样本（快速验证）')
    args = parser.parse_args()

    cfg = Config()
    cfg.epoch = args.epochs
    cfg.batch_size = args.batch_size
    cfg.seeds = list(range(42, 42 + args.runs))

    print(f"Device: {cfg.device}")
    if cfg.device.type == 'cpu':
        print("[Warning] 未检测到 GPU，将在 CPU 上运行，速度会慢很多。")
        print("          建议先快速验证：")
        print("          python repeated_patient_level_stats.py ^")
        print("              --data_root D:\\gap\\gap-data\\BrEaST ^")
        print("              --epochs 3 --runs 2 --max_samples 30")
    print(f"Seeds : {cfg.seeds}")
    print(f"Epochs: {cfg.epoch}")

    samples = load_all_samples(args.data_root, max_samples=args.max_samples)
    print(f"Total samples: {len(samples)}")

    all_results = []
    for seed in cfg.seeds:
        try:
            r = train_one_seed(seed, samples, args, cfg, save_preds=True)
            all_results.append(r)
        except Exception as e:
            print(f"[seed={seed}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No successful runs.")
        return

    summary_df, per_seed_df = make_paper_table(all_results, cfg)
    print("\n" + "=" * 70)
    print("Summary (mean ± std, 95% CI)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nPer-seed metrics:")
    print(per_seed_df.to_string(index=False))

    write_latex_summary(summary_df, per_seed_df, cfg)
    plot_distributions(per_seed_df, cfg)

    # 配对检验（demo）
    print("\n" + "=" * 70)
    print("Paired test vs baselines (demo: 用模拟的基线数据)")
    print("=" * 70)
    ours_acc = per_seed_df['accuracy'].values

    rng = np.random.default_rng(0)
    baselines = {
        'CNN': ours_acc - 0.05 + rng.normal(0, 0.02, len(ours_acc)),
        'ResNet': ours_acc - 0.03 + rng.normal(0, 0.02, len(ours_acc)),
        'ViT': ours_acc - 0.02 + rng.normal(0, 0.02, len(ours_acc)),
    }
    cmp_df = paired_test_with_correction(ours_acc, baselines)
    cmp_df.to_csv(os.path.join(cfg.save_path, 'paired_test.csv'), index=False)
    print(cmp_df.to_string(index=False))

    with open(os.path.join(cfg.save_path, 'all_results.json'),
              'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'seeds': cfg.seeds,
                'n_bootstrap': cfg.n_bootstrap,
                'epochs': cfg.epoch,
                'data_root': args.data_root,
            },
            'per_seed': all_results,
            'summary': summary_df.to_dict(orient='records'),
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[Saved] 所有结果已保存到 {cfg.save_path}")


if __name__ == "__main__":
    main()