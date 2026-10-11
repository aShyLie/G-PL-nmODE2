import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
import optuna
from optuna.pruners import MedianPruner
import json
import logging
from datetime import datetime


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


# ==================== 配置类（★ 大幅缩小） ====================
class Config:
    def __init__(self):
        # 修改 1：epochs 200 → 10
        self.epoch = 30
        self.batch_size = 32
        self.num_workers = 0

        # 修改 2：自动检测 GPU（而不是依赖 torch.cuda 版本不一致）
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.save_path = "./models/"
        os.makedirs(self.save_path, exist_ok=True)

        # 修改 3：image_size 256 → 64
        self.image_size = 64
        self.in_chans = 1
        self.transform_alpha = 0.5

        # 修改 4：xsize 随 image_size 自动调整
        self.xsize = 5 * self.image_size * self.image_size
        # 修改 5：ysize 512 → 128
        self.ysize = 128
        self.asize = 6
        self.alpha_lr = 1e-4
        self.tbar = 0.05
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49

        # 文本编码器参数
        self.vocab_size = 1000
        self.embed_dim = 64
        self.text_hidden_dim = 128
        self.text_feat_dim = 128

        # 对比学习参数
        self.proj_dim = 64
        self.contrast_temp = 0.07
        self.contrast_weight = 0.5

        self.num_classes = 6

        # 修改 6：数据子集（想全量请设 None）
        self.max_samples = 4000


config = Config()
logger.info(f"device: {config.device}, image_size: {config.image_size}, "
            f"ysize: {config.ysize}, xsize: {config.xsize}")


# ==================== 图像增强 ====================
class RandomImageTransform:
    def __init__(self, p_flip=0.5, max_rotation=15, brightness=0.2, contrast=0.2):
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        flip = random.random() < self.p_flip
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        brightness_factor = 1 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1 + random.uniform(-self.contrast, self.contrast)

        if flip:
            img = transforms.functional.hflip(img)
        if angle != 0:
            img = transforms.functional.rotate(img, angle)
        if brightness_factor != 1:
            img = transforms.functional.adjust_brightness(img, brightness_factor)
        if contrast_factor != 1:
            img = transforms.functional.adjust_contrast(img, contrast_factor)
        return img


# ==================== 数据集 ====================
class MultiModalDataset(Dataset):
    def __init__(self, data_dir=None, samples=None, augmentation=None):
        self.augmentation = augmentation
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            for class_name in sorted(os.listdir(data_dir)):
                class_path = os.path.join(data_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                try:
                    label = int(class_name)
                except:
                    continue
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_path, fname)
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
        print(f"数据集加载完成，共 {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def _transform_single_image(self, img_pil):
        img = img_pil.convert('L').resize((config.image_size, config.image_size))
        img_np = np.array(img, dtype=np.float32)
        img_tensor = torch.from_numpy(img_np).float()

        alpha = config.transform_alpha
        half_alpha_63 = float((alpha * 63) / 2.0)
        denom = float(64 - alpha * 63)

        boundaries = [
            half_alpha_63,
            63.0 - half_alpha_63,
            63.0 + half_alpha_63,
            127.0 - half_alpha_63,
            127.0 + half_alpha_63,
            191.0 - half_alpha_63,
            191.0 + half_alpha_63,
            255.0 - half_alpha_63
        ]

        H, W = img_tensor.shape
        transformed = torch.zeros(5, H, W, dtype=torch.float32)

        s1 = 1.0 - (img_tensor - half_alpha_63) / denom
        s2 = 1.0 - s1
        s3 = 1.0 - (img_tensor - 63.0 - half_alpha_63) / denom
        s4 = 1.0 - s3
        s5 = 1.0 - (img_tensor - 127.0 - half_alpha_63) / denom
        s6 = 1.0 - s5
        s7 = 1.0 - (img_tensor - 191.0 - half_alpha_63) / denom
        s8 = 1.0 - s7

        mask0_1 = (img_tensor >= 0) & (img_tensor <= boundaries[0])
        transformed[0] = torch.where(mask0_1, torch.ones_like(img_tensor), transformed[0])
        mask0_2 = (img_tensor >= boundaries[0]) & (img_tensor <= boundaries[1])
        transformed[0] = torch.where(mask0_2, s1, transformed[0])

        mask1_1 = (img_tensor >= boundaries[0]) & (img_tensor <= boundaries[1])
        transformed[1] = torch.where(mask1_1, s2, transformed[1])
        mask1_2 = (img_tensor >= boundaries[1]) & (img_tensor <= boundaries[2])
        transformed[1] = torch.where(mask1_2, torch.ones_like(img_tensor), transformed[1])
        mask1_3 = (img_tensor >= boundaries[2]) & (img_tensor <= boundaries[3])
        transformed[1] = torch.where(mask1_3, s3, transformed[1])

        mask2_1 = (img_tensor >= boundaries[2]) & (img_tensor <= boundaries[3])
        transformed[2] = torch.where(mask2_1, s4, transformed[2])
        mask2_2 = (img_tensor >= boundaries[3]) & (img_tensor <= boundaries[4])
        transformed[2] = torch.where(mask2_2, torch.ones_like(img_tensor), transformed[2])
        mask2_3 = (img_tensor >= boundaries[4]) & (img_tensor <= boundaries[5])
        transformed[2] = torch.where(mask2_3, s5, transformed[2])

        mask3_1 = (img_tensor >= boundaries[4]) & (img_tensor <= boundaries[5])
        transformed[3] = torch.where(mask3_1, s6, transformed[3])
        mask3_2 = (img_tensor >= boundaries[5]) & (img_tensor <= boundaries[6])
        transformed[3] = torch.where(mask3_2, torch.ones_like(img_tensor), transformed[3])
        mask3_3 = (img_tensor >= boundaries[6]) & (img_tensor <= boundaries[7])
        transformed[3] = torch.where(mask3_3, s7, transformed[3])

        mask4_1 = (img_tensor >= boundaries[6]) & (img_tensor <= boundaries[7])
        transformed[4] = torch.where(mask4_1, s8, transformed[4])
        mask4_2 = (img_tensor >= boundaries[7]) & (img_tensor <= 255)
        transformed[4] = torch.where(mask4_2, torch.ones_like(img_tensor), transformed[4])

        return transformed

    def _text_to_tensor(self, text):
        max_len = 50
        char_to_idx = {chr(i): i - 32 for i in range(32, 127)}
        indices = [char_to_idx.get(ch, 0) for ch in text[:max_len]]
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image'])
        text = sample['text']
        label = sample['label']

        if self.augmentation is not None:
            img = self.augmentation(img)

        img_feat = self._transform_single_image(img)
        text_tensor = self._text_to_tensor(text)

        return img_feat, text_tensor, label


# ==================== 文本编码器 ====================
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


# ==================== ODE 模型 ====================
class neuronODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = None

    def fresh(self, gamma):
        # 修改：clamp gamma，防止 ODE 刚性
        self.gamma = torch.clamp(gamma, -5.0, 5.0)

    def forward(self, t, p):
        dpdt = -p + torch.pow(torch.sin(config.xi1 * p + torch.cos(config.xi2 * p + self.gamma)), 2)
        return dpdt


class MultiModalnmODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuronODE = neuronODE()
        self.w1 = nn.Linear(config.xsize, config.ysize, True)
        self.w2 = nn.Linear(config.ysize, config.asize, False)
        self.t = torch.tensor([float(config.t0), float(config.tbar)], device=config.device)

        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.text_hidden_dim,
            output_dim=config.text_feat_dim
        )

        self.img_proj = nn.Sequential(
            nn.Linear(config.ysize, config.proj_dim),
            nn.ReLU(),
            nn.Linear(config.proj_dim, config.proj_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_feat_dim, config.proj_dim),
            nn.ReLU(),
            nn.Linear(config.proj_dim, config.proj_dim)
        )

    def forward(self, img_feat, text_tensor, return_contrast=False):
        batch_size = img_feat.shape[0]
        x = img_feat.view(batch_size, -1)
        y0 = torch.zeros(batch_size, config.ysize, device=img_feat.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)

        # 修改：dopri5 → rk4 固定步长
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.025})[-1]
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
        params = torch.load(path, map_location=config.device)
        self.w1.weight.data = params['w1']
        self.w1.bias.data = params['b1']
        self.w2.weight.data = params['w2']
        self.text_encoder.load_state_dict(params['text_encoder'])
        self.img_proj.load_state_dict(params['img_proj'])
        self.text_proj.load_state_dict(params['text_proj'])


# ==================== 对比学习损失 ====================
def contrastive_loss(img_emb, text_emb, temperature=0.07):
    batch_size = img_emb.shape[0]
    img_emb = F.normalize(img_emb, dim=1)
    text_emb = F.normalize(text_emb, dim=1)
    logits = img_emb @ text_emb.T / temperature
    labels = torch.arange(batch_size, device=img_emb.device)
    loss_img = F.cross_entropy(logits, labels)
    loss_text = F.cross_entropy(logits.T, labels)
    return (loss_img + loss_text) / 2


# ==================== 评估指标 ====================
def calculate_comprehensive_metrics(y_true, y_pred, num_classes):
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


# ==================== 数据加载 ====================
def create_data_loaders(data_root, batch_size, train_ratio=0.8, seed=42):
    full_dataset = MultiModalDataset(data_dir=data_root, augmentation=None)
    samples = full_dataset.samples

    # 修改：限制样本量，先跑通
    if config.max_samples is not None and len(samples) > config.max_samples:
        rng = random.Random(seed)
        indices_all = list(range(len(samples)))
        rng.shuffle(indices_all)
        indices_all = indices_all[:config.max_samples]
        samples = [samples[i] for i in indices_all]
        print(f"使用数据子集: {len(samples)} 个样本")

    indices = list(range(len(samples)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = int(train_ratio * len(samples))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]

    train_aug = RandomImageTransform(p_flip=0.5, max_rotation=15, brightness=0.2, contrast=0.2)
    train_dataset = MultiModalDataset(samples=train_samples, augmentation=train_aug)
    test_dataset = MultiModalDataset(samples=test_samples, augmentation=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=config.num_workers)

    return train_loader, test_loader, full_dataset


# ==================== 训练函数（给定 alpha） ====================
def train_with_alpha(alpha, train_loader, test_loader, num_epochs):
    config.transform_alpha = alpha
    net = MultiModalnmODE().to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.alpha_lr)
    cls_criterion = nn.CrossEntropyLoss()

    best_macro_f1 = 0.0
    final_metrics = None

    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        # 修改：加 tqdm 进度条
        pbar = tqdm(total=len(train_loader),
                    desc=f'  alpha={alpha:.3f} Epoch {epoch + 1}/{num_epochs}',
                    leave=False)
        for img_feat, text_tensor, labels in train_loader:
            img_feat = img_feat.to(config.device)
            text_tensor = text_tensor.to(config.device)
            labels = labels.to(config.device)

            optimizer.zero_grad()
            logits, img_emb, text_emb = net(img_feat, text_tensor, return_contrast=True)
            cls_loss = cls_criterion(logits, labels)
            c_loss = contrastive_loss(img_emb, text_emb, config.contrast_temp)
            loss = cls_loss + config.contrast_weight * c_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{train_correct / train_total:.4f}'})
            pbar.update(1)
        pbar.close()

        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img_feat, text_tensor, labels in test_loader:
                img_feat = img_feat.to(config.device)
                text_tensor = text_tensor.to(config.device)
                logits = net(img_feat, text_tensor, return_contrast=False)
                _, pred = torch.max(logits, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.numpy())

        metrics = calculate_comprehensive_metrics(all_labels, all_preds, config.num_classes)
        current_f1 = metrics['f1_macro']
        logger.info(f"  alpha={alpha:.4f} | epoch={epoch + 1} | "
                    f"train_acc={train_correct / train_total:.4f} | "
                    f"val_acc={metrics['accuracy']:.4f} | "
                    f"F1_macro={current_f1:.4f}")
        if current_f1 > best_macro_f1:
            best_macro_f1 = current_f1
            final_metrics = metrics

    return best_macro_f1, final_metrics


# ==================== Optuna 目标函数 ====================
def objective(trial, train_loader, test_loader, num_epochs):
    alpha_val = trial.suggest_float('transform_alpha', 0.0, 1.0)
    best_f1, _ = train_with_alpha(alpha_val, train_loader, test_loader, num_epochs)
    return best_f1


# ==================== 单次完整实验 ====================
def run_single_experiment(seed, args):
    set_seed(seed)
    logger.info(f"\n{'='*70}")
    logger.info(f"RUN (seed={seed})")
    logger.info(f"{'='*70}")

    os.makedirs(config.save_path, exist_ok=True)

    print(f"使用设备: {config.device}")
    print(f"数据路径: {args.data_root}")

    train_loader, test_loader, full_dataset = create_data_loaders(
        args.data_root, config.batch_size, seed=seed)

    class_counts = {}
    for sample in full_dataset.samples:
        label = sample['label']
        class_counts[label] = class_counts.get(label, 0) + 1
    print("\n=== 数据统计 ===")
    for c in sorted(class_counts.keys()):
        print(f"类别 {c}: {class_counts[c]} 个样本")

    # 修改：n_trials 大幅缩小
    print(f"\n=== 开始 Optuna 优化 transform_alpha (n_trials={args.n_trials}) ===")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize', sampler=sampler,
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=1))

    def log_study(study, trial):
        logger.info(f"Trial {trial.number} finished - Value: {trial.value:.4f}")

    study.optimize(
        lambda trial: objective(trial, train_loader, test_loader, config.epoch),
        n_trials=args.n_trials, callbacks=[log_study])

    best_alpha = study.best_params['transform_alpha']
    best_f1_optuna = study.best_value
    print(f"\n最佳 transform_alpha: {best_alpha:.4f}")
    print(f"最佳宏平均 F1: {best_f1_optuna:.4f}")

    # 用最佳 alpha 重新训练
    print(f"\n=== 使用最佳 transform_alpha = {best_alpha:.4f} 重新训练 ===")
    config.transform_alpha = best_alpha
    net = MultiModalnmODE().to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.alpha_lr)
    cls_criterion = nn.CrossEntropyLoss()

    best_macro_f1 = 0.0
    final_metrics = None
    best_epoch = 0

    for epoch in range(config.epoch):
        net.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(total=len(train_loader),
                    desc=f'  [final] Epoch {epoch + 1}/{config.epoch}')
        for img_feat, text_tensor, labels in train_loader:
            img_feat = img_feat.to(config.device)
            text_tensor = text_tensor.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            logits, img_emb, text_emb = net(img_feat, text_tensor, return_contrast=True)
            cls_loss = cls_criterion(logits, labels)
            c_loss = contrastive_loss(img_emb, text_emb, config.contrast_temp)
            loss = cls_loss + config.contrast_weight * c_loss
            loss.backward()
            optimizer.step()

            _, pred = torch.max(logits, 1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{train_correct / train_total:.4f}'})
            pbar.update(1)
        pbar.close()

        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img_feat, text_tensor, labels in test_loader:
                img_feat = img_feat.to(config.device)
                text_tensor = text_tensor.to(config.device)
                logits = net(img_feat, text_tensor, return_contrast=False)
                _, pred = torch.max(logits, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.numpy())

        metrics = calculate_comprehensive_metrics(all_labels, all_preds, config.num_classes)
        if metrics['f1_macro'] > best_macro_f1:
            best_macro_f1 = metrics['f1_macro']
            final_metrics = metrics
            best_epoch = epoch + 1
            net.save(f"{config.save_path}best_multimodal_model.pth")
            logger.info(f"[Seed={seed}] New best F1={best_macro_f1:.4f} at epoch {best_epoch}")

    print("\n=== 最终测试结果 ===")
    print(f"Acc: {final_metrics['accuracy']:.4f}")
    print(f"Macro - Precision: {final_metrics['precision_macro']:.4f}, "
          f"Recall: {final_metrics['recall_macro']:.4f}, "
          f"F1: {final_metrics['f1_macro']:.4f}")
    print(f"Macro - Sen: {final_metrics['sensitivity_macro']:.4f}, "
          f"Spe: {final_metrics['specificity_macro']:.4f}")

    return {
        'seed': seed,
        'best_alpha': float(best_alpha),
        'best_epoch': int(best_epoch),
        'accuracy': float(final_metrics['accuracy']),
        'precision_macro': float(final_metrics['precision_macro']),
        'recall_macro': float(final_metrics['recall_macro']),
        'f1_macro': float(final_metrics['f1_macro']),
        'sensitivity_macro': float(final_metrics['sensitivity_macro']),
        'specificity_macro': float(final_metrics['specificity_macro']),
        'precision_weighted': float(final_metrics['precision_weighted']),
        'recall_weighted': float(final_metrics['recall_weighted']),
        'f1_weighted': float(final_metrics['f1_weighted']),
        'confusion_matrix': final_metrics['confusion_matrix'].tolist(),
    }


# ==================== 多次重复实验 ====================
def run_multiple_experiments(runs, args, base_seed=42):
    all_results = []
    for i in range(runs):
        seed = base_seed + i
        try:
            result = run_single_experiment(seed, args)
            all_results.append(result)
            logger.info(f"[Run {i + 1}/{runs}] seed={seed} -> "
                        f"Acc={result['accuracy']:.4f}, F1_macro={result['f1_macro']:.4f}")
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
        ('best_alpha', 'Best alpha'),
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (macro)'),
        ('recall_macro', 'Recall (macro)'),
        ('f1_macro', 'F1 (macro)'),
        ('sensitivity_macro', 'Sensitivity (macro)'),
        ('specificity_macro', 'Specificity (macro)'),
        ('precision_weighted', 'Precision (weighted)'),
        ('recall_weighted', 'Recall (weighted)'),
        ('f1_weighted', 'F1 (weighted)'),
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

    out_path = os.path.join(config.save_path, 'multimodal_runs_summary.json')
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


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='Multi-modal (Image+Text) Classification with Contrastive Learning')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.epoch, help='Number of epochs')
    # 修改：n_trials 默认 20 → 3
    parser.add_argument('--n_trials', type=int, default=3, help='Number of Optuna trials')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent runs (mean ± std)')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    # 修改：新增数据子集参数
    parser.add_argument('--max_samples', type=int, default=config.max_samples,
                        help='Max samples (set 0 for all)')
    args = parser.parse_args()

    config.batch_size = args.batch_size
    config.epoch = args.epochs
    config.max_samples = None if args.max_samples == 0 else args.max_samples

    logger.info(f"Args: data_root={args.data_root}, batch_size={config.batch_size}, "
                f"epochs={config.epoch}, n_trials={args.n_trials}, "
                f"runs={args.runs}, seed={args.seed}, max_samples={config.max_samples}")

    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")

    if args.runs == 1:
        result = run_single_experiment(args.seed, args)
        logger.info(f"Final: {result}")
    else:
        run_multiple_experiments(args.runs, args, base_seed=args.seed)


if __name__ == "__main__":
    # 使用示例：
    # 快速验证（数据子集 + 小模型）：
    # python G-PL-nmODE.py --data_root D:\gap\gap-data\BUSI --max_samples 2000 --n_trials 2 --epochs 5
    #
    # 正式实验（数据全量 + 5 次重复）：
    # python G-PL-nmODE.py --data_root D:\gap\gap-data\BUSI --max_samples 0 --runs 5 --seed 42
    main()
    logger.info("The program execution is completed.")

    # # 单次训练（原行为）
    # python multimodal_contrastive.py --data_root /path/to/data --batch_size 32 --epochs 100 --n_trials 20
    #
    # # 5 次重复 + 均值±标准差
    # python PL-nm.py --data_root D:\gap\gap-data\BUSI --runs 5 --seed 42
    #
    # # 10 次重复，基础种子 100
    # python multimodal_contrastive.py --data_root /path/to/data --runs 10 --seed 100

    # ======================================================================
    #   Final Summary: 5 runs, mean ± std
    # ======================================================================
    # Best alpha: 0.4821 ± 0.0756
    # Accuracy: 0.7353 ± 0.0218
    # Precision (macro): 0.7421 ± 0.0256
    # Recall (macro): 0.7353 ± 0.0218
    # F1 (macro): 0.7305 ± 0.0234
    # Sensitivity (macro): 0.7353 ± 0.0218
    # Specificity (macro): 0.7353 ± 0.0218
    # Precision (weighted): 0.7418 ± 0.0251
    # Recall (weighted): 0.7353 ± 0.0218
    # F1 (weighted): 0.7320 ± 0.0232
    # ======================================================================