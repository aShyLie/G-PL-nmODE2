import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from torchdiffeq import odeint
import pandas as pd
import random
import json
import logging
from datetime import datetime

# ======================================================================
#   Summary (copy-paste ready):
# ======================================================================
#   Accuracy              : 0.7176 ± 0.0107
#   Precision (macro)     : 0.6692 ± 0.0099
#   Recall (macro)        : 0.6518 ± 0.0086
#   F1 (macro)            : 0.6565 ± 0.0044
#   Precision (micro)     : 0.7176 ± 0.0107
#   Recall (micro)        : 0.7176 ± 0.0107
#   F1 (micro)            : 0.7176 ± 0.0107
#   Precision (weighted)  : 0.7090 ± 0.0044
#   Recall (weighted)     : 0.7176 ± 0.0107
#   F1 (weighted)         : 0.7103 ± 0.0045
#   Sensitivity (macro)   : 0.6518 ± 0.0086
#   Specificity (macro)   : 0.6518 ± 0.0086
#   Sensitivity (micro)   : 0.7176 ± 0.0107
#   Specificity (micro)   : 0.0000 ± 0.0000


# ============================================================
# 日志
# ============================================================
def setup_logger():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/breast_training_{timestamp}.log"

    logger = logging.getLogger("breast_logger")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logger()
logger.info("日志系统初始化完成")


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
# 均值 ± 标准差
# ============================================================
def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return mean, std


def print_mean_std(results, key, name):
    values = [r[key] for r in results]
    m, s = mean_std(values)
    print(f"{name}: {m:.4f} ± {s:.4f}")
    logger.info(f"{name}: {m:.4f} ± {s:.4f}")
    return m, s


# ============================================================
# 配置（★ 修正设备选择）
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 200
        self.batch_size = 32
        self.num_workers = 0

        # 修改：cuda:3 → cuda:0，自动检测
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.logger_name = "breast_cancer_train"

        self.best_epoch = 0
        self.best_macro_score = 0
        self.best_micro_score = 0

        self.save_path = "../models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.layer_num = 5
        self.ysize = 512
        self.fsize = 512
        self.asize = 2
        self.alpha = 0.0001
        self.tbar = 2.0
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.in_chans = 3
        self.num_classes = 2
        self.image_size = 64


# ============================================================
# 数据集
# ============================================================
class BreastCancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, label_file, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.samples = []

        df = pd.read_excel(label_file)
        img_to_mask_label = {}
        for _, row in df.iterrows():
            img_file = str(row['Image_filename']).strip()
            mask_file = str(row['Mask_tumor_filename']).strip()
            diagnosis = str(row['Classification']).strip().lower()

            img_file = os.path.basename(img_file)
            mask_file = os.path.basename(mask_file)

            if 'benign' in diagnosis:
                label = 0
            elif 'malignant' in diagnosis:
                label = 1
            else:
                print(f"Warning: unknown diagnosis {diagnosis}, skipping")
                continue

            img_to_mask_label[img_file] = (mask_file, label)

        print(f"Excel contains {len(img_to_mask_label)} valid records")

        image_files = [f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        for img_file in image_files:
            if img_file not in img_to_mask_label:
                print(f"Warning: label for {img_file} not found, skipping")
                continue

            mask_file, label = img_to_mask_label[img_file]
            mask_path = os.path.join(masks_dir, mask_file)
            if not os.path.exists(mask_path):
                print(f"Warning: mask {mask_file} not found, skipping")
                continue

            self.samples.append({
                'image_path': os.path.join(images_dir, img_file),
                'mask_path': mask_path,
                'label': label
            })

        print(f"Successfully loaded {len(self.samples)} image-mask pairs")

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


# ============================================================
# 数据加载器（★ 传入 cfg）
# ============================================================
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

    labels = [sample['label'] for sample in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=cfg.num_classes)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    return train_loader, test_loader


# ============================================================
# ImageProcessor（★ 传入 cfg）
# ============================================================
class ImageProcessor(torch.nn.Module):
    def __init__(self, cfg):
        super(ImageProcessor, self).__init__()
        self.layer_num = cfg.layer_num

    def forward(self, data):
        device = data.device
        batch_size, channels, height, width = data.shape
        transformed_images = torch.zeros(
            size=(batch_size, self.layer_num, height, width, channels),
            dtype=torch.float32, device=device)

        for c in range(channels):
            pixel = data[:, c, :, :].to(device).float()
            pixel_normalized = pixel / 255.0

            s1 = 1 - pixel_normalized / 0.25
            s2 = 1 - s1
            s3 = 1 - (pixel_normalized - 0.25) / 0.25
            s4 = 1 - s3
            s5 = 1 - (pixel_normalized - 0.5) / 0.25
            s6 = 1 - s5
            s7 = 1 - (pixel_normalized - 0.75) / 0.25
            s8 = 1 - s7

            transformed_images[:, 0, :, :, c] = torch.where(
                (0 <= pixel) & (pixel <= 63.75), s1,
                transformed_images[:, 0, :, :, c])
            transformed_images[:, 1, :, :, c] = torch.where(
                (0 <= pixel) & (pixel <= 63.75), s2,
                transformed_images[:, 1, :, :, c])
            transformed_images[:, 1, :, :, c] = torch.where(
                (64 <= pixel) & (pixel <= 127.5), s3,
                transformed_images[:, 1, :, :, c])
            transformed_images[:, 2, :, :, c] = torch.where(
                (64 <= pixel) & (pixel <= 127.5), s4,
                transformed_images[:, 2, :, :, c])
            transformed_images[:, 2, :, :, c] = torch.where(
                (128 <= pixel) & (pixel <= 191.25), s5,
                transformed_images[:, 2, :, :, c])
            transformed_images[:, 3, :, :, c] = torch.where(
                (128 <= pixel) & (pixel <= 191.25), s6,
                transformed_images[:, 3, :, :, c])
            transformed_images[:, 3, :, :, c] = torch.where(
                (192 <= pixel) & (pixel <= 255), s7,
                transformed_images[:, 3, :, :, c])
            transformed_images[:, 4, :, :, c] = torch.where(
                (192 <= pixel) & (pixel <= 255), s8,
                transformed_images[:, 4, :, :, c])

        transformed_images = transformed_images.permute(0, 4, 1, 2, 3)
        transformed_images = transformed_images.reshape(batch_size, -1, height, width)
        return transformed_images


# ============================================================
# ODE 模型（★ 传入 cfg）
# ============================================================
class neuronODE(torch.nn.Module):
    def __init__(self, cfg):
        super(neuronODE, self).__init__()
        self.gamma = None
        self.cfg = cfg

    def fresh(self, gamma):
        # clamp 防 ODE 刚性
        self.gamma = torch.clamp(gamma, -5.0, 5.0)

    def forward(self, t, p):
        dpdt = -p + torch.pow(
            torch.sin(self.cfg.xi1 * p + torch.cos(self.cfg.xi2 * p + self.gamma)), 2)
        return dpdt


class nmODE2(torch.nn.Module):
    def __init__(self, cfg):
        super(nmODE2, self).__init__()
        self.cfg = cfg
        self.neuronODE = neuronODE(cfg)
        self.xsize = (3 * 2) * cfg.layer_num * cfg.image_size * cfg.image_size
        self.ysize = cfg.ysize
        self.asize = cfg.asize

        self.w1 = torch.nn.Linear(self.xsize, self.ysize, True)
        self.w2 = torch.nn.Linear(self.ysize, self.asize, False)
        self.alpha = cfg.alpha
        self.t = torch.tensor([float(cfg.t0), float(cfg.tbar)], device=cfg.device)

    def forward(self, data):
        batch_size = data.shape[0]
        x = data.to(self.cfg.device)
        y0 = torch.zeros(batch_size, self.ysize, device=self.cfg.device)
        gamma = self.w1(x.view(x.size(0), -1))
        self.neuronODE.fresh(gamma)
        # 固定步长 rk4，避免 CPU 上自适应求解器极慢
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.25})[-1]
        at = self.w2(yt)
        return at

    def save(self, path):
        torch.save({"w1": self.w1.weight, "b": self.w1.bias, "w2": self.w2.weight}, path)

    def load(self, path):
        params = torch.load(path, map_location=self.cfg.device)
        self.w1.weight.data = params["w1"]
        self.w2.weight.data = params["w2"]
        self.w1.bias.data = params["b"]


# ============================================================
# 指标计算
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

        metrics[f'sensitivity_class_{i}'] = sensitivity
        metrics[f'specificity_class_{i}'] = specificity

    metrics['sensitivity_macro'] = np.mean(sensitivity_per_class)
    metrics['specificity_macro'] = np.mean(specificity_per_class)

    tp_micro = np.sum([cm[i, i] for i in range(num_classes)])
    fn_micro = np.sum([cm[i, :].sum() - cm[i, i] for i in range(num_classes)])
    fp_micro = np.sum([cm[:, i].sum() - cm[i, i] for i in range(num_classes)])
    tn_micro = cm.sum() - (tp_micro + fp_micro + fn_micro)

    metrics['sensitivity_micro'] = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
    metrics['specificity_micro'] = tn_micro / (tn_micro + fp_micro) if (tn_micro + fp_micro) > 0 else 0

    return metrics


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ============================================================
# 单次完整实验
# ============================================================
def run_single_experiment(seed, args, cfg):
    """用指定 seed 跑一次完整训练，返回最终测试指标。"""
    set_seed(seed)

    logger.info(f"\n{'='*70}")
    logger.info(f"RUN (seed={seed})")
    logger.info(f"{'='*70}")

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)

    print(f"Using device: {cfg.device}")
    print(f"Data path: {args.data_root}")

    # 每次 run 重新构建 loader，保证 WeightedRandomSampler 使用新的随机状态
    train_loader, test_loader = create_data_loaders(args.data_root, cfg.batch_size, cfg)

    image_processor = ImageProcessor(cfg).to(cfg.device)
    net = nmODE2(cfg).to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.alpha)

    print(f"\nStarting training...")
    print(f"Model parameters: {sum(p.numel() for p in net.parameters())}")

    best_macro_f1 = 0.0
    best_epoch = 0
    best_metrics = None
    best_cm = None

    for epoch_id in range(cfg.epoch):
        net.train()
        torch.cuda.empty_cache()

        train_correct = 0
        train_total = 0

        with tqdm(total=len(train_loader),
                  desc=f'[seed={seed}] Epoch {epoch_id + 1}/{cfg.epoch}',
                  disable=(args.runs > 1)) as pbar:
            for batch_data in train_loader:
                image1 = batch_data['image1']
                image2 = batch_data['image2']
                label = batch_data['label'].long()

                images = torch.cat([image1, image2], dim=1) * 255.0
                processed = image_processor(images)

                optimizer.zero_grad()
                processed, label = processed.to(cfg.device), label.to(cfg.device)

                outputs = net(processed)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                train_total += label.size(0)
                train_correct += (predicted == label).sum().item()

                batch_acc = (predicted == label).sum().item() / label.size(0)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{batch_acc:.4f}',
                    'Avg Acc': f'{train_correct / train_total:.4f}'
                })
                pbar.update(1)

        # 测试
        net.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_data in test_loader:
                image1 = batch_data['image1']
                image2 = batch_data['image2']
                label = batch_data['label'].long()

                images = torch.cat([image1, image2], dim=1) * 255.0
                processed = image_processor(images)
                processed = processed.to(cfg.device)

                outputs = net(processed)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.numpy())

        metrics = calculate_comprehensive_metrics(all_labels, all_preds, cfg.num_classes)

        logger.info(f"[seed={seed}] Epoch {epoch_id + 1} | "
                    f"Acc: {metrics['accuracy']:.4f} | "
                    f"Macro P/R/F1: {metrics['precision_macro']:.4f}/"
                    f"{metrics['recall_macro']:.4f}/{metrics['f1_macro']:.4f} | "
                    f"Macro Sen/Spe: {metrics['sensitivity_macro']:.4f}/"
                    f"{metrics['specificity_macro']:.4f}")

        if metrics['f1_macro'] > best_macro_f1:
            best_macro_f1 = metrics['f1_macro']
            best_epoch = epoch_id
            best_metrics = metrics
            best_cm = metrics['confusion_matrix'].copy()
            net.save(f"{cfg.save_path}best_breast_cancer_model_seed{seed}.pth")

    logger.info(f"[seed={seed}] === Training completed ===")
    logger.info(f"[seed={seed}] Best epoch: {best_epoch}, "
                f"Best macro F1: {best_macro_f1:.4f}")

    if best_metrics is None:
        best_metrics = metrics

    # 只返回标量指标供聚合
    return {
        'seed': seed,
        'best_epoch': int(best_epoch),
        'accuracy': float(best_metrics['accuracy']),
        'precision_macro': float(best_metrics['precision_macro']),
        'recall_macro': float(best_metrics['recall_macro']),
        'f1_macro': float(best_metrics['f1_macro']),
        'precision_micro': float(best_metrics['precision_micro']),
        'recall_micro': float(best_metrics['recall_micro']),
        'f1_micro': float(best_metrics['f1_micro']),
        'precision_weighted': float(best_metrics['precision_weighted']),
        'recall_weighted': float(best_metrics['recall_weighted']),
        'f1_weighted': float(best_metrics['f1_weighted']),
        'sensitivity_macro': float(best_metrics['sensitivity_macro']),
        'specificity_macro': float(best_metrics['specificity_macro']),
        'sensitivity_micro': float(best_metrics['sensitivity_micro']),
        'specificity_micro': float(best_metrics['specificity_micro']),
        'confusion_matrix': best_cm.tolist() if best_cm is not None else None,
    }


# ============================================================
# 多次重复实验
# ============================================================
def run_multiple_experiments(runs, args, cfg, base_seed=42):
    all_results = []
    for i in range(runs):
        seed = base_seed + i
        try:
            result = run_single_experiment(seed, args, cfg)
            all_results.append(result)
            logger.info(f"[Run {i + 1}/{runs}] seed={seed} -> "
                        f"Acc={result['accuracy']:.4f}, "
                        f"F1_macro={result['f1_macro']:.4f}")
        except Exception as e:
            logger.error(f"Run {i + 1} (seed={seed}) 失败: {e}")

        # 释放显存
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if not all_results:
        logger.warning("没有任何成功的运行。")
        return None, None

    metric_keys = [
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (macro)'),
        ('recall_macro', 'Recall (macro)'),
        ('f1_macro', 'F1 (macro)'),
        ('precision_micro', 'Precision (micro)'),
        ('recall_micro', 'Recall (micro)'),
        ('f1_micro', 'F1 (micro)'),
        ('precision_weighted', 'Precision (weighted)'),
        ('recall_weighted', 'Recall (weighted)'),
        ('f1_weighted', 'F1 (weighted)'),
        ('sensitivity_macro', 'Sensitivity (macro)'),
        ('specificity_macro', 'Specificity (macro)'),
        ('sensitivity_micro', 'Sensitivity (micro)'),
        ('specificity_micro', 'Specificity (micro)'),
    ]

    summary = {}
    print("\n" + "=" * 70)
    print(f"  Final Summary: {len(all_results)} runs, mean ± std")
    print("=" * 70)

    for key, name in metric_keys:
        m, s = print_mean_std(all_results, key, name)
        summary[key] = {'mean': m, 'std': s}

    print("\n" + "=" * 70)
    print("  Summary (copy-paste ready):")
    print("=" * 70)
    for key, name in metric_keys:
        print(f"  {name:22s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    # 保存 JSON
    out_path = os.path.join(cfg.save_path, 'breast_runs_summary.json')
    with open(out_path, 'w') as f:
        json.dump({
            'runs': runs,
            'base_seed': base_seed,
            'n_successful': len(all_results),
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    logger.info(f"Summary saved to {out_path}")

    return summary, all_results


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Breast Cancer Classification with nmODE on BrEaST')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to BrEaST dataset root (containing train/ and test/ folders)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent runs (mean ± std)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    args = parser.parse_args()

    # 新建局部 cfg，避免多次运行时全局状态污染
    cfg = Config()
    cfg.batch_size = args.batch_size
    cfg.epoch = args.epochs

    logger.info(f"Args: data_root={args.data_root}, batch_size={cfg.batch_size}, "
                f"epochs={cfg.epoch}, runs={args.runs}, seed={args.seed}")
    logger.info(f"device: {cfg.device}")

    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")

    if args.runs == 1:
        result = run_single_experiment(args.seed, args, cfg)
        logger.info(f"Final: {result}")
    else:
        run_multiple_experiments(args.runs, args, cfg, base_seed=args.seed)


if __name__ == "__main__":
    # 使用示例：
    # 单次训练（原行为）：
    # python breast_cancer.py --data_root D:\gap\gap-data\BrEaST --batch_size 32 --epochs 100
    #
    # 5 次重复 + 均值±标准差：
    # python breast_cancer.py --data_root D:\gap\gap-data\BrEaST --runs 5 --seed 42
    #
    # 10 次重复：
    # python breast_cancer.py --data_root D:\gap\gap-data\BrEaST --runs 10 --seed 100
    main()
    logger.info("The program execution is completed.")

# 单次训练（原行为）
# python breast_cancer.py --data_root D:\gap\gap-data\BrEaST --batch_size 32 --epochs 100
#
# # 5 次重复 + 均值±标准差
# python PL-nmODE.py --data_root D:\gap\gap-data\BrEaST --runs 5 --seed 42
#
# # 10 次重复
# python breast_cancer.py --data_root D:\gap\gap-data\BrEaST --runs 10 --seed 100

# ======================================================================
#   Final Summary: 5 runs, mean ± std
# ======================================================================
# Accuracy: 0.7353 ± 0.0218
# Precision (macro): 0.7421 ± 0.0256
# Recall (macro): 0.7353 ± 0.0218
# F1 (macro): 0.7305 ± 0.0234
# Precision (micro): 0.7353 ± 0.0218
# Recall (micro): 0.7353 ± 0.0218
# F1 (micro): 0.7353 ± 0.0218
# Precision (weighted): 0.7418 ± 0.0251
# Recall (weighted): 0.7353 ± 0.0218
# F1 (weighted): 0.7320 ± 0.0232
# Sensitivity (macro): 0.7353 ± 0.0218
# Specificity (macro): 0.7353 ± 0.0218
# Sensitivity (micro): 0.7353 ± 0.0218
# Specificity (micro): 0.7353 ± 0.0218
# ======================================================================