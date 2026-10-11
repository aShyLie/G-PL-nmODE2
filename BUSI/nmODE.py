import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from torchdiffeq import odeint
import random
import json
import logging
from datetime import datetime

# ======================================================================
#   Summary (copy-paste ready):
# ======================================================================
#   Accuracy              : 0.5181 ± 0.0048
#   Precision (macro)     : 0.0864 ± 0.0008
#   Recall (macro)        : 0.1667 ± 0.0000
#   F1 (macro)            : 0.1138 ± 0.0007
#   Precision (micro)     : 0.5181 ± 0.0048
#   Recall (micro)        : 0.5181 ± 0.0048
#   F1 (micro)            : 0.5181 ± 0.0048
#   Precision (weighted)  : 0.2685 ± 0.0049
#   Recall (weighted)     : 0.5181 ± 0.0048


# ============================================================
# 日志
# ============================================================
def setup_logger():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/multimodal_{timestamp}.log"

    logger = logging.getLogger("multimodal_logger")
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
# 均值 ± 标准差工具
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
# 配置类（改为实例属性，避免多 run 污染）
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 30
        self.batch_size = 32
        self.num_workers = 0

        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
            torch.cuda.set_device(1)
        else:
            self.device = torch.device("cpu")

        self.logger_name = "breast_cancer_multimodal"

        self.best_epoch = 0
        self.best_macro_score = 0
        self.best_micro_score = 0

        self.save_path = "./models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.layer_num = 5
        self.xsize = 3 * 256 * 256
        self.ysize = 1024
        self.fsize = 1024 * 5
        self.asize = 7
        self.alpha = 0.0001
        self.tbar = 0.01
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.in_chans = 3
        self.num_classes = 6
        self.image_size = 256

        # 文本编码器
        self.vocab_size = 128
        self.embed_dim = 64
        self.text_hidden_dim = 128
        self.text_feat_dim = 128

        # 对比学习
        self.proj_dim = 128
        self.contrast_temp = 0.07
        self.contrast_weight = 0.5

        self.use_class_weight = True


config = Config()
logger.info(f"device: {config.device}")


# ============================================================
# 数据集类
# ============================================================
class BreastCancerDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []

        for folder in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                label = int(folder)
            except ValueError:
                continue

            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(folder_path, fname)
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                    else:
                        text = f"这是一张类别为 {label} 的医学图像。"
                    self.samples.append({
                        'image': img_path,
                        'text': text,
                        'label': label
                    })

    def __len__(self):
        return len(self.samples)

    def _text_to_tensor(self, text):
        max_len = 50
        indices = [ord(ch) % 128 for ch in text[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image']).convert('RGB')
        text = sample['text']
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        text_tensor = self._text_to_tensor(text)
        return {'image': image, 'text': text_tensor, 'label': label}


# ============================================================
# 文本编码器
# ============================================================
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        pooled = out[:, -1, :]
        feat = self.fc(pooled)
        return feat


# ============================================================
# ODE 模型
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


class MultiModalnmODE(torch.nn.Module):
    def __init__(self, cfg):
        super(MultiModalnmODE, self).__init__()
        self.cfg = cfg
        self.neuronODE = neuronODE(cfg)
        self.w1 = torch.nn.Linear(cfg.xsize, cfg.ysize, True)
        self.w2 = torch.nn.Linear(cfg.ysize, cfg.num_classes, False)
        self.t = torch.tensor([float(cfg.t0), float(cfg.tbar)], device=cfg.device)

        self.text_encoder = TextEncoder(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.text_hidden_dim,
            output_dim=cfg.text_feat_dim
        )

        self.img_proj = nn.Sequential(
            nn.Linear(cfg.ysize, cfg.proj_dim),
            nn.ReLU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_feat_dim, cfg.proj_dim),
            nn.ReLU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim)
        )

    def forward(self, img, text_tensor, return_contrast=False):
        batch_size = img.shape[0]
        x = img.view(batch_size, -1)
        y0 = torch.zeros(batch_size, self.cfg.ysize, device=img.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)
        # 固定步长 rk4，避免自适应求解器在 CPU 上极慢
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.0025})[-1]
        logits = self.w2(yt)

        if not return_contrast:
            return logits

        img_emb = self.img_proj(gamma)
        text_feat = self.text_encoder(text_tensor)
        text_emb = self.text_proj(text_feat)
        return logits, img_emb, text_emb

    def save(self, path):
        torch.save({
            'w1': self.w1.weight, 'b1': self.w1.bias,
            'w2': self.w2.weight,
            'text_encoder': self.text_encoder.state_dict(),
            'img_proj': self.img_proj.state_dict(),
            'text_proj': self.text_proj.state_dict()
        }, path)

    def load(self, path):
        params = torch.load(path, map_location=self.cfg.device)
        self.w1.weight.data = params['w1']
        self.w1.bias.data = params['b1']
        self.w2.weight.data = params['w2']
        self.text_encoder.load_state_dict(params['text_encoder'])
        self.img_proj.load_state_dict(params['img_proj'])
        self.text_proj.load_state_dict(params['text_proj'])


# ============================================================
# 对比学习损失
# ============================================================
def contrastive_loss(img_emb, text_emb, temperature=0.07):
    batch_size = img_emb.shape[0]
    img_emb = F.normalize(img_emb, dim=1)
    text_emb = F.normalize(text_emb, dim=1)
    logits = img_emb @ text_emb.T / temperature
    labels = torch.arange(batch_size, device=img_emb.device)
    loss_img = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels)
    return (loss_img + loss_text) / 2


# ============================================================
# 评估指标
# ============================================================
def calculate_comprehensive_metrics(y_true, y_pred, num_classes=6):
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

    sensitivity_per_class, specificity_per_class = [], []
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
# 数据加载器创建（★ 修正原代码的 is_train 污染）
# ============================================================
def create_data_loaders(data_root, cfg, batch_size=32, train_ratio=0.8, seed=42):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = BreastCancerDataset(data_root, transform=transform, is_train=True)
    samples = full_dataset.samples
    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = int(train_ratio * len(samples))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]

    # 用 __new__ 构造独立 dataset，避免 random_split 共享导致的 is_train 污染
    train_dataset = BreastCancerDataset.__new__(BreastCancerDataset)
    train_dataset.data_dir = data_root
    train_dataset.transform = transform
    train_dataset.is_train = True
    train_dataset.samples = train_samples

    test_dataset = BreastCancerDataset.__new__(BreastCancerDataset)
    test_dataset.data_dir = data_root
    test_dataset.transform = transform
    test_dataset.is_train = False
    test_dataset.samples = test_samples

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=cfg.num_workers)
    return train_loader, test_loader, full_dataset


# ============================================================
# 单次完整实验
# ============================================================
def run_single_experiment(seed, args, cfg):
    """用指定 seed 跑一次完整训练，返回最佳指标字典。"""
    set_seed(seed)

    logger.info(f"\n{'='*70}")
    logger.info(f"RUN (seed={seed})")
    logger.info(f"{'='*70}")

    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)

    print(f"使用设备: {cfg.device}")
    print(f"数据路径: {args.data_root}")

    train_loader, test_loader, full_dataset = create_data_loaders(
        args.data_root, cfg, cfg.batch_size, seed=seed)

    # 类别分布
    class_counts = {}
    for sample in full_dataset.samples:
        label = sample['label']
        class_counts[label] = class_counts.get(label, 0) + 1
    print("\n=== 数据统计 ===")
    for c in sorted(class_counts.keys()):
        print(f"类别 {c}: {class_counts[c]} 个样本")

    # 模型
    net = MultiModalnmODE(cfg).to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.alpha)

    print(f"\n开始训练...")
    print(f"模型参数数量: {sum(p.numel() for p in net.parameters())}")

    patience = 50
    patience_counter = 0
    best_accuracy = 0.0
    best_macro_f1 = 0.0
    best_epoch = 0
    best_metrics = None

    for epoch_id in range(cfg.epoch):
        net.train()
        torch.cuda.empty_cache()

        train_correct = 0
        train_total = 0
        total_loss = 0.0

        with tqdm(total=len(train_loader),
                  desc=f'[seed={seed}] Epoch {epoch_id+1}/{cfg.epoch}',
                  disable=(args.runs > 1)) as pbar:
            for batch_data in train_loader:
                image = batch_data['image'].to(cfg.device)
                text = batch_data['text'].to(cfg.device)
                label = batch_data['label'].long().to(cfg.device)

                optimizer.zero_grad()

                logits, img_emb, text_emb = net(image, text, return_contrast=True)
                cls_loss = criterion(logits, label)
                c_loss = contrastive_loss(img_emb, text_emb, cfg.contrast_temp)
                loss = cls_loss + cfg.contrast_weight * c_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, pred = torch.max(logits, 1)
                train_correct += (pred == label).sum().item()
                train_total += label.size(0)
                batch_acc = (pred == label).sum().item() / label.size(0)

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'ClsLoss': f'{cls_loss.item():.4f}',
                    'ContLoss': f'{c_loss.item():.4f}',
                    'Acc': f'{batch_acc:.4f}',
                    'AvgAcc': f'{train_correct/train_total:.4f}'
                })
                pbar.update(1)

        # 测试
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_data in test_loader:
                image = batch_data['image'].to(cfg.device)
                text = batch_data['text'].to(cfg.device)
                label = batch_data['label'].long().to(cfg.device)

                logits = net(image, text, return_contrast=False)
                _, pred = torch.max(logits, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        metrics = calculate_comprehensive_metrics(all_labels, all_preds, cfg.num_classes)
        current_acc = metrics['accuracy']
        current_f1 = metrics['f1_macro']

        logger.info(f"[seed={seed}] Epoch {epoch_id+1} | "
                    f"Acc={current_acc:.4f} | "
                    f"Macro P/R/F1: {metrics['precision_macro']:.4f}/"
                    f"{metrics['recall_macro']:.4f}/{current_f1:.4f} | "
                    f"Sen/Spe: {metrics['sensitivity_macro']:.4f}/"
                    f"{metrics['specificity_macro']:.4f}")

        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            best_accuracy = current_acc
            best_epoch = epoch_id
            best_metrics = metrics
            patience_counter = 0
            net.save(f"{cfg.save_path}best_multimodal_model_seed{seed}.pth")
            logger.info(f"[seed={seed}] New best macro F1={best_macro_f1:.4f} "
                        f"at epoch {epoch_id}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"[seed={seed}] Early stopping at epoch {epoch_id+1}")
            break

    logger.info(f"[seed={seed}] === Training completed ===")
    logger.info(f"[seed={seed}] Best epoch: {best_epoch}, "
                f"Best acc: {best_accuracy:.4f}, Best macro F1: {best_macro_f1:.4f}")

    if best_metrics is None:
        best_metrics = metrics

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
        'confusion_matrix': best_metrics['confusion_matrix'].tolist(),
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

    out_path = os.path.join(cfg.save_path, 'multimodal_runs_summary.json')
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
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Multi-modal (Image+Text) Classification with Contrastive Learning')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.epoch, help='Number of epochs')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent runs (mean ± std)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    args = parser.parse_args()

    # 新建局部 cfg，避免多次运行时全局状态污染
    cfg = Config()
    cfg.batch_size = args.batch_size
    cfg.epoch = args.epochs

    logger.info(f"Args: data_root={args.data_root}, batch_size={cfg.batch_size}, "
                f"epochs={cfg.epoch}, runs={args.runs}, seed={args.seed}, device={cfg.device}")

    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")

    if args.runs == 1:
        result = run_single_experiment(args.seed, args, cfg)
        logger.info(f"Final: {result}")
        # 绘制最终混淆矩阵
        cm = np.array(result['confusion_matrix'])
        plot_confusion_matrix(
            cm, [f'Class_{i}' for i in range(cfg.num_classes)],
            f'Final Confusion Matrix (seed={args.seed})',
            save_path=os.path.join(cfg.save_path, f'confusion_matrix_seed{args.seed}.png'))
    else:
        run_multiple_experiments(args.runs, args, cfg, base_seed=args.seed)


if __name__ == "__main__":
    # 使用示例：
    # 单次训练：
    # python multimodal_contrastive.py --data_root /path/to/data --batch_size 32 --epochs 100
    #
    # 5 次重复 + 均值±标准差：
    # python nmODE.py --data_root D:\gap\gap-data\BUSI --runs 5 --seed 42
    #
    # 10 次重复：
    # python multimodal_contrastive.py --data_root /path/to/data --runs 10 --seed 100
    main()
    logger.info("The program execution is completed.")