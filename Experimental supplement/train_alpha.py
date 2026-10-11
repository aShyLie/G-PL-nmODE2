import os
import json
import glob
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
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
# 配置（★ α 学习率大幅提升）
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 100                 # ★ 从 50 提升到 100
        self.batch_size = 32
        self.num_workers = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.save_path = "../models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.ysize = 1024
        self.asize = 2
        self.alpha_lr = 1e-4              # ★ 主网络学习率：从 1e-5 提升到 1e-4
        self.tbar = 1.0
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.num_classes = 2
        self.image_size = 128

        # ★ α 参数
        self.alpha_min = 0.1
        self.alpha_max = 0.9
        self.alpha_init_raw = 0.0         # σ(0)=0.5 → 初始 α=0.5
        self.smooth_k = 20.0              # ★ 从 10 提升到 20，让梯度更明显
        self.alpha_param_lr = 5e-3        # ★ α 的独立学习率（直接指定）
        self.alpha_warmup_epochs = 10     # α 的学习率 warmup


config = Config()


# ============================================================
# 可学习 ImageProcessor（STE + Sigmoid 平滑）
# ============================================================
class ImageProcessor(nn.Module):
    def __init__(self, alpha_min=0.1, alpha_max=0.9,
                 alpha_init_raw=0.0, smooth_k=20.0):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.smooth_k = smooth_k
        self.raw_alpha = nn.Parameter(torch.tensor(alpha_init_raw, dtype=torch.float32))

    def get_alpha(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)

    def _soft_gate(self, pixel, b_lo, b_hi):
        k = self.smooth_k
        return torch.sigmoid(k * (pixel - b_lo)) * torch.sigmoid(k * (b_hi - pixel))

    def forward(self, x):
        B, C, H, W = x.shape
        if x.dtype != torch.float32:
            x = x.float()
        device = x.device

        alpha = self.get_alpha()
        alpha_63 = alpha * 63.0
        half_alpha_63 = alpha_63 / 2.0

        b0 = half_alpha_63
        b1 = 63.0 - half_alpha_63
        b2 = 63.0 + half_alpha_63
        b3 = 127.0 - half_alpha_63
        b4 = 127.0 + half_alpha_63
        b5 = 191.0 - half_alpha_63
        b6 = 191.0 + half_alpha_63
        b7 = 255.0 - half_alpha_63

        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        one = torch.tensor(1.0, dtype=torch.float32, device=device)
        max_val = torch.tensor(255.0, dtype=torch.float32, device=device)
        c64 = torch.tensor(64.0, dtype=torch.float32, device=device)
        c63 = torch.tensor(63.0, dtype=torch.float32, device=device)
        c127 = torch.tensor(127.0, dtype=torch.float32, device=device)
        c191 = torch.tensor(191.0, dtype=torch.float32, device=device)

        out = torch.zeros(B, C * 5, H, W, dtype=torch.float32, device=device)

        for c in range(C):
            pixel = x[:, c:c + 1, :, :]

            s1 = one - (pixel - b0) / (c64 - alpha_63)
            s2 = one - s1
            s3 = one - (pixel - c63 - b0) / (c64 - alpha_63)
            s4 = one - s3
            s5 = one - (pixel - c127 - b0) / (c64 - alpha_63)
            s6 = one - s5
            s7 = one - (pixel - c191 - b0) / (c64 - alpha_63)
            s8 = one - s7

            # Level 0
            m_hard_0 = torch.where(
                (pixel >= zero) & (pixel <= b0), one,
                torch.where((pixel >= b0) & (pixel <= b1), s1, zero))
            g_0a = self._soft_gate(pixel, zero, b0)
            g_0b = self._soft_gate(pixel, b0, b1)
            m_soft_0 = g_0a * one + g_0b * s1
            out[:, c * 5 + 0] = (m_hard_0.detach() + m_soft_0 - m_soft_0.detach()).squeeze(1)

            # Level 1
            m_hard_1 = torch.where(
                (pixel >= b0) & (pixel <= b1), s2,
                torch.where((pixel >= b1) & (pixel <= b2), one,
                            torch.where((pixel >= b2) & (pixel <= b3), s3, zero)))
            g_1a = self._soft_gate(pixel, b0, b1)
            g_1b = self._soft_gate(pixel, b1, b2)
            g_1c = self._soft_gate(pixel, b2, b3)
            m_soft_1 = g_1a * s2 + g_1b * one + g_1c * s3
            out[:, c * 5 + 1] = (m_hard_1.detach() + m_soft_1 - m_soft_1.detach()).squeeze(1)

            # Level 2
            m_hard_2 = torch.where(
                (pixel >= b2) & (pixel <= b3), s4,
                torch.where((pixel >= b3) & (pixel <= b4), one,
                            torch.where((pixel >= b4) & (pixel <= b5), s5, zero)))
            g_2a = self._soft_gate(pixel, b2, b3)
            g_2b = self._soft_gate(pixel, b3, b4)
            g_2c = self._soft_gate(pixel, b4, b5)
            m_soft_2 = g_2a * s4 + g_2b * one + g_2c * s5
            out[:, c * 5 + 2] = (m_hard_2.detach() + m_soft_2 - m_soft_2.detach()).squeeze(1)

            # Level 3
            m_hard_3 = torch.where(
                (pixel >= b4) & (pixel <= b5), s6,
                torch.where((pixel >= b5) & (pixel <= b6), one,
                            torch.where((pixel >= b6) & (pixel <= b7), s7, zero)))
            g_3a = self._soft_gate(pixel, b4, b5)
            g_3b = self._soft_gate(pixel, b5, b6)
            g_3c = self._soft_gate(pixel, b6, b7)
            m_soft_3 = g_3a * s6 + g_3b * one + g_3c * s7
            out[:, c * 5 + 3] = (m_hard_3.detach() + m_soft_3 - m_soft_3.detach()).squeeze(1)

            # Level 4
            m_hard_4 = torch.where(
                (pixel >= b6) & (pixel <= b7), s8,
                torch.where((pixel >= b7) & (pixel <= max_val), one, zero))
            g_4a = self._soft_gate(pixel, b6, b7)
            g_4b = self._soft_gate(pixel, b7, max_val)
            m_soft_4 = g_4a * s8 + g_4b * one
            out[:, c * 5 + 4] = (m_hard_4.detach() + m_soft_4 - m_soft_4.detach()).squeeze(1)

        return out


# ============================================================
# 数据集
# ============================================================
class BreastCancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, label_file, transform=None):
        self.transform = transform
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

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        for img_file in image_files:
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        mask = Image.open(sample['mask_path']).convert('RGB')
        label = sample['label']
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return {'image1': image, 'image2': mask, 'label': label}


def create_data_loaders(data_root, batch_size=32, cfg=None):
    if cfg is None:
        cfg = Config()

    train_images_dir = os.path.join(data_root, 'train', 'Images')
    train_masks_dir = os.path.join(data_root, 'train', 'Masks')
    test_images_dir = os.path.join(data_root, 'test', 'Images')
    test_masks_dir = os.path.join(data_root, 'test', 'Masks')
    label_file = os.path.join(data_root, 'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx')

    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = BreastCancerDataset(train_images_dir, train_masks_dir, label_file, transform)
    test_dataset = BreastCancerDataset(test_images_dir, test_masks_dir, label_file, transform)

    labels = [s['label'] for s in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=cfg.num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return train_loader, test_loader


# ============================================================
# ODE 模型
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
        self.input_channels = 6 * 5
        self.xsize = self.input_channels * cfg.image_size * cfg.image_size
        self.ysize = cfg.ysize
        self.asize = cfg.num_classes

        self.w1 = nn.Linear(self.xsize, self.ysize, True)
        self.w2 = nn.Linear(self.ysize, self.asize, False)
        self.t = torch.tensor([float(cfg.t0), float(cfg.tbar)], device=cfg.device)

    def forward(self, data):
        B = data.shape[0]
        x = data.to(self.cfg.device).view(B, -1)
        y0 = torch.zeros(B, self.ysize, device=self.cfg.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.25})[-1]
        return self.w2(yt)

    def save(self, path):
        torch.save({"w1": self.w1.weight, "b": self.w1.bias, "w2": self.w2.weight}, path)

    def load(self, path):
        params = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.w1.weight.data = params["w1"]
        self.w2.weight.data = params["w2"]
        self.w1.bias.data = params["b"]


# ============================================================
# 指标
# ============================================================
def calculate_comprehensive_metrics(y_true, y_pred, num_classes=2):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro'], _ = \
        precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'], metrics['recall_micro'], metrics['f1_micro'], _ = \
        precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted'], _ = \
        precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    metrics['confusion_matrix'] = cm

    sensitivity_per_class = []
    specificity_per_class = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity_per_class.append(sensitivity)
        specificity_per_class.append(specificity)

    metrics['sensitivity_macro'] = np.mean(sensitivity_per_class)
    metrics['specificity_macro'] = np.mean(specificity_per_class)
    return metrics


# ============================================================
# 单次训练（★ α 独立学习率 + warmup）
# ============================================================
def run_single_experiment(seed, args, cfg):
    set_seed(seed)

    print(f"\n{'='*70}")
    print(f"RUN (seed={seed})")
    print(f"{'='*70}")
    print(f"Using device: {cfg.device}")

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)

    train_loader, test_loader = create_data_loaders(args.data_root, cfg.batch_size, cfg)

    image_processor = ImageProcessor(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        alpha_init_raw=cfg.alpha_init_raw,
        smooth_k=cfg.smooth_k,
    ).to(cfg.device)

    net = nmODE2(cfg).to(cfg.device)
    criterion = nn.CrossEntropyLoss().to(cfg.device)

    # ★ α 使用独立的大学习率
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': cfg.alpha_lr},
        {'params': [image_processor.raw_alpha], 'lr': cfg.alpha_param_lr},
    ])

    # ★ α 的 warmup：前 N 个 epoch 线性增长
    def alpha_lr_lambda(epoch):
        if epoch < cfg.alpha_warmup_epochs:
            return (epoch + 1) / cfg.alpha_warmup_epochs
        return 1.0

    # 只对 raw_α 生效的 warmup（通过手动设置 param_group 的 lr）
    alpha_param_lr_base = cfg.alpha_param_lr

    print(f"Model params: {sum(p.numel() for p in net.parameters())}")
    print(f"Learnable α: raw_α = {image_processor.raw_alpha.item():.4f}, "
          f"α = {image_processor.get_alpha().item():.4f}")

    best_macro_f1 = 0.0
    best_epoch = 0
    best_metrics = None

    alpha_history = []
    train_loss_history = []
    val_f1_history = []

    for epoch_id in range(cfg.epoch):
        # ★ 手动设置 α 的 warmup
        warmup_scale = alpha_lr_lambda(epoch_id)
        optimizer.param_groups[1]['lr'] = alpha_param_lr_base * warmup_scale

        net.train()
        image_processor.train()
        torch.cuda.empty_cache()

        train_correct = 0
        train_total = 0
        epoch_loss = 0.0

        with tqdm(total=len(train_loader),
                  desc=f'[seed={seed}] Epoch {epoch_id+1}/{cfg.epoch}',
                  disable=(args.runs > 1)) as pbar:
            for batch_data in train_loader:
                image1 = batch_data['image1']
                image2 = batch_data['image2']
                label = batch_data['label'].long()

                images = torch.cat([image1, image2], dim=1) * 255.0
                processed = image_processor(images)

                optimizer.zero_grad()
                processed = processed.to(cfg.device)
                label = label.to(cfg.device)

                outputs = net(processed)
                loss = criterion(outputs, label)
                loss.backward()

                # ★ 对 α 的梯度做裁剪
                if image_processor.raw_alpha.grad is not None:
                    image_processor.raw_alpha.grad.data.clamp_(-0.1, 0.1)

                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{train_correct/train_total:.4f}',
                    'α': f'{image_processor.get_alpha().item():.4f}',
                })
                pbar.update(1)

        current_alpha = image_processor.get_alpha().item()
        alpha_history.append(current_alpha)
        train_loss_history.append(epoch_loss / len(train_loader))

        # 测试
        net.eval()
        image_processor.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                image1 = batch_data['image1']
                image2 = batch_data['image2']
                label = batch_data['label'].long()
                images = torch.cat([image1, image2], dim=1) * 255.0
                processed = image_processor(images).to(cfg.device)
                outputs = net(processed)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.numpy())

        metrics = calculate_comprehensive_metrics(all_labels, all_preds, cfg.num_classes)
        val_f1_history.append(metrics['f1_macro'])

        print(f"Epoch {epoch_id+1:3d} | α={current_alpha:.4f} | "
              f"Acc={metrics['accuracy']:.4f} | "
              f"F1_macro={metrics['f1_macro']:.4f} | "
              f"P={metrics['precision_macro']:.4f} | "
              f"R={metrics['recall_macro']:.4f} | "
              f"lr_α={optimizer.param_groups[1]['lr']:.5f}")

        if metrics['f1_macro'] > best_macro_f1:
            best_macro_f1 = metrics['f1_macro']
            best_epoch = epoch_id
            best_metrics = metrics
            net.save(f"{cfg.save_path}best_breast_cancer_model_seed{seed}.pth")
            torch.save({
                'raw_alpha': image_processor.raw_alpha.data,
                'alpha': image_processor.get_alpha().data,
                'epoch': epoch_id,
            }, f"{cfg.save_path}best_alpha_seed{seed}.pth")

    print(f"\n[seed={seed}] Best epoch={best_epoch}, Best macro F1={best_macro_f1:.4f}")
    print(f"[seed={seed}] Final α = {alpha_history[-1]:.4f}")
    print(f"[seed={seed}] α history (每 5 个 epoch): "
          f"{[round(a, 4) for a in alpha_history[::5]]}")

    if best_metrics is None:
        best_metrics = metrics

    return {
        'seed': seed,
        'best_epoch': int(best_epoch),
        'final_alpha': float(alpha_history[-1]),
        'alpha_history': [float(a) for a in alpha_history],
        'train_loss_history': [float(l) for l in train_loss_history],
        'val_f1_history': [float(f) for f in val_f1_history],
        'accuracy': float(best_metrics['accuracy']),
        'precision_macro': float(best_metrics['precision_macro']),
        'recall_macro': float(best_metrics['recall_macro']),
        'f1_macro': float(best_metrics['f1_macro']),
        'sensitivity_macro': float(best_metrics['sensitivity_macro']),
        'specificity_macro': float(best_metrics['specificity_macro']),
    }


# ============================================================
# 多次重复
# ============================================================
def run_multiple_experiments(runs, args, cfg, base_seed=42):
    all_results = []
    for i in range(runs):
        seed = base_seed + i
        try:
            result = run_single_experiment(seed, args, cfg)
            all_results.append(result)
        except Exception as e:
            print(f"Run {i+1} (seed={seed}) failed: {e}")
            import traceback
            traceback.print_exc()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if not all_results:
        print("No successful runs.")
        return None

    final_alphas = [r['final_alpha'] for r in all_results]
    mean_alpha = float(np.mean(final_alphas))
    std_alpha = float(np.std(final_alphas, ddof=1)) if len(final_alphas) > 1 else 0.0

    print("\n" + "=" * 70)
    print(f"  α 收敛结果: {mean_alpha:.4f} ± {std_alpha:.4f}  (n={len(final_alphas)})")
    print("=" * 70)
    for i, a in enumerate(final_alphas):
        print(f"  seed={base_seed+i}: final_α = {a:.4f}")

    print("\n" + "=" * 70)
    print("  其他指标")
    print("=" * 70)
    metric_keys = [
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (macro)'),
        ('recall_macro', 'Recall (macro)'),
        ('f1_macro', 'F1 (macro)'),
        ('sensitivity_macro', 'Sensitivity (macro)'),
        ('specificity_macro', 'Specificity (macro)'),
    ]
    summary = {}
    for key, name in metric_keys:
        vals = np.array([r[key] for r in all_results], dtype=float)
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary[key] = {'mean': m, 'std': s}
        print(f"  {name:22s}: {m:.4f} ± {s:.4f}")

    # ★ 绘制 α 收敛曲线（y 轴自适应）
    plt.figure(figsize=(9, 5))
    for r in all_results:
        plt.plot(range(1, len(r['alpha_history']) + 1),
                 r['alpha_history'],
                 alpha=0.8, linewidth=2, label=f"seed={r['seed']}")
    plt.axhline(y=mean_alpha, color='red', linestyle='--',
                label=f'mean={mean_alpha:.4f}±{std_alpha:.4f}')
    plt.axhline(y=cfg.alpha_min, color='gray', linestyle=':', alpha=0.5,
                label=f'α_min={cfg.alpha_min}')
    plt.axhline(y=cfg.alpha_max, color='gray', linestyle=':', alpha=0.5,
                label=f'α_max={cfg.alpha_max}')
    plt.xlabel('Epoch')
    plt.ylabel(r'Learned $\alpha$')
    plt.title(r'Convergence of learnable $\alpha$ across different seeds')
    plt.legend(loc='best', fontsize=8)
    plt.grid(alpha=0.3)
    # ★ y 轴自动缩放，方便看到变化趋势
    all_alpha = [a for r in all_results for a in r['alpha_history']]
    y_min, y_max = min(all_alpha), max(all_alpha)
    y_pad = max(0.01, (y_max - y_min) * 0.5)
    plt.ylim(max(cfg.alpha_min - 0.05, y_min - y_pad),
             min(cfg.alpha_max + 0.05, y_max + y_pad))
    plt.tight_layout()
    out_fig = os.path.join(cfg.save_path, 'alpha_convergence.png')
    plt.savefig(out_fig, dpi=200)
    plt.close()
    print(f"\n[Saved] α 收敛曲线: {out_fig}")

    # 保存 JSON
    out_json = os.path.join(cfg.save_path, 'alpha_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'runs': runs,
            'base_seed': base_seed,
            'final_alpha_mean': mean_alpha,
            'final_alpha_std': std_alpha,
            'final_alphas': final_alphas,
            'metrics_summary': summary,
            'per_run': all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[Saved] 全部结果: {out_json}")

    # LaTeX 表
    tex_lines = []
    tex_lines.append("% ===== Table: Learnable alpha convergence =====")
    tex_lines.append("\\begin{table}[h]\\centering")
    tex_lines.append("\\caption{Convergence of learnable $\\alpha$ across "
                     + str(runs) + " random seeds.}")
    tex_lines.append("\\begin{tabular}{lc}\\hline")
    tex_lines.append("Metric & Value \\\\ \\hline")
    tex_lines.append("Final $\\alpha$ (mean $\\pm$ std) & "
                     + f"${mean_alpha:.4f} \\pm {std_alpha:.4f}$" + " \\\\")
    tex_lines.append("$\\alpha_{\\min}$ & " + f"${cfg.alpha_min:.2f}$" + " \\\\")
    tex_lines.append("$\\alpha_{\\max}$ & " + f"${cfg.alpha_max:.2f}$" + " \\\\")
    tex_lines.append("$N$ seeds & " + str(runs) + " \\\\")
    tex_lines.append("\\hline\\end{tabular}\\end{table}")

    with open(os.path.join(cfg.save_path, 'alpha_table.tex'), 'w', encoding='utf-8') as f:
        f.write("\n".join(tex_lines))
    print(f"[Saved] LaTeX 表: {os.path.join(cfg.save_path, 'alpha_table.tex')}")

    return summary


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--epochs', type=int, default=config.epoch)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.batch_size = args.batch_size
    cfg.epoch = args.epochs

    if args.runs == 1:
        r = run_single_experiment(args.seed, args, cfg)
        print(f"\nFinal α = {r['final_alpha']:.4f}")
    else:
        run_multiple_experiments(args.runs, args, cfg, base_seed=args.seed)


if __name__ == "__main__":
    main()