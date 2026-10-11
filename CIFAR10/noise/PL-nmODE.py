import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
import cv2
import random
import logging
import json
from datetime import datetime

# ======================================================================
#   Summary (copy-paste ready):
# ======================================================================
#   Test Accuracy         : 0.5711 ± 0.0031
#   Test Loss             : 1.2262 ± 0.0085
#   Best Test Accuracy    : 0.5711 ± 0.0031
#
# Saved to ./models/multi_seed_summary.json

# ============================================================
# 日志系统
# ============================================================
def setup_logger():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/training_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()
logger.info("日志系统初始化完成")


# ============================================================
# 统一随机种子设置
# ============================================================
def set_seed(seed):
    """统一设置所有随机种子，保证重复实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
# 配置
# ============================================================
class Config:
    epoch = 1000
    batch_size = 128
    num_workers = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger_name = "train"
    best_epoch = 0
    best_acc = 0
    save_path = "./models/"
    xsize = 32 * 32
    ysize = 256
    asize = 10
    alpha = 0.0001
    tbar = 1.0
    t0 = 0
    xi1 = 0.5
    xi2 = 0.5
    in_chans = 3
    num_classes = 10

    layer_num = 5

    # ===== 噪声配置（替代原模糊配置）=====
    # 可用噪声类型: 'gaussian' | 'salt_pepper' | 'poisson' | 'speckle'
    noise_types = ['gaussian', 'salt_pepper', 'poisson', 'speckle']
    noise_params = [0.02, 0.05, 0.10]   # 与噪声类型匹配的强度参数
    noise_type = 'salt_pepper'          # 默认噪声（与第二个代码一致）
    noise_param = 0.02

    early_stopping_patience = 30
    early_stopping_min_delta = 0.0001
    early_stopping_counter = 0


config = Config()


# ============================================================
# ImageProcessor（保持不变）
# ============================================================
class ImageProcessor(torch.nn.Module):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.layer_num = config.layer_num

    def forward(self, data):
        batch_size, channels, height, width = data.shape
        assert channels == 3, f"期望3通道输入，但得到{channels}通道"

        if data.dtype != torch.float32:
            data = data.to(torch.float32)

        transformed_images = torch.zeros(
            size=(batch_size, 3 * self.layer_num, height, width),
            dtype=torch.float32, device=data.device
        )

        data_scaled = (data * 255.0).to(torch.float32)

        for channel in range(3):
            pixel = data_scaled[:, channel, :, :]

            s1 = (1.0 - pixel / 64.0).to(torch.float32)
            s2 = (1.0 - s1).to(torch.float32)
            s3 = (1.0 - (pixel - 63.0) / 64.0).to(torch.float32)
            s4 = (1.0 - s3).to(torch.float32)
            s5 = (1.0 - (pixel - 127.0) / 64.0).to(torch.float32)
            s6 = (1.0 - s5).to(torch.float32)
            s7 = (1.0 - (pixel - 191.0) / 64.0).to(torch.float32)
            s8 = (1.0 - s7).to(torch.float32)

            s1 = torch.clamp(s1, 0.0, 1.0)
            s2 = torch.clamp(s2, 0.0, 1.0)
            s3 = torch.clamp(s3, 0.0, 1.0)
            s4 = torch.clamp(s4, 0.0, 1.0)
            s5 = torch.clamp(s5, 0.0, 1.0)
            s6 = torch.clamp(s6, 0.0, 1.0)
            s7 = torch.clamp(s7, 0.0, 1.0)
            s8 = torch.clamp(s8, 0.0, 1.0)

            mask_0_63 = ((pixel >= 0) & (pixel <= 63)).to(torch.float32)
            mask_64_127 = ((pixel >= 64) & (pixel <= 127)).to(torch.float32)
            mask_128_191 = ((pixel >= 128) & (pixel <= 191)).to(torch.float32)
            mask_192_255 = ((pixel >= 192) & (pixel <= 255)).to(torch.float32)

            base_idx = channel * self.layer_num

            transformed_images[:, base_idx + 0, :, :] = torch.where(
                mask_0_63.bool(), s1,
                torch.tensor(0.0, dtype=torch.float32, device=data.device))

            layer2_temp = torch.where(
                mask_0_63.bool(), s2,
                torch.tensor(0.0, dtype=torch.float32, device=data.device))
            transformed_images[:, base_idx + 1, :, :] = torch.where(
                mask_64_127.bool(), s3, layer2_temp)

            layer3_temp = torch.where(
                mask_64_127.bool(), s4,
                torch.tensor(0.0, dtype=torch.float32, device=data.device))
            transformed_images[:, base_idx + 2, :, :] = torch.where(
                mask_128_191.bool(), s5, layer3_temp)

            layer4_temp = torch.where(
                mask_128_191.bool(), s6,
                torch.tensor(0.0, dtype=torch.float32, device=data.device))
            transformed_images[:, base_idx + 3, :, :] = torch.where(
                mask_192_255.bool(), s7, layer4_temp)

            transformed_images[:, base_idx + 4, :, :] = torch.where(
                mask_192_255.bool(), s8,
                torch.tensor(0.0, dtype=torch.float32, device=data.device))

        return transformed_images


# ============================================================
# 噪声变换（替代原 BlurTransform）
# ------------------------------------------------------------
# 参考第二个代码中的噪声函数实现：
#   - add_gaussian_noise     : 高斯噪声
#   - add_salt_and_pepper_noise : 椒盐噪声
#   - add_poisson_noise      : 泊松噪声
#   - add_speckle_noise      : 斑点噪声
# 作用对象：ToTensor() 之后的 [0,1] 张量
# ============================================================
class NoiseTransform(object):
    def __init__(self, noise_type='salt_pepper', noise_param=0.02, use_mixed_noise=True):
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.use_mixed_noise = use_mixed_noise

    # ---------- 各类噪声实现 ----------
    @staticmethod
    def add_gaussian_noise(image, mean=0.0, std=0.1):
        """高斯噪声"""
        noise = torch.randn_like(image) * std + mean
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    @staticmethod
    def add_salt_and_pepper_noise(image, prob=0.02):
        """椒盐噪声"""
        noisy_image = image.clone()
        rand_vals = torch.rand_like(noisy_image)
        noisy_image[rand_vals < prob / 2] = 0.0   # 椒（黑）
        noisy_image[rand_vals > 1 - prob / 2] = 1.0  # 盐（白）
        return noisy_image

    @staticmethod
    def add_poisson_noise(image, scale=1.0):
        """泊松噪声"""
        noisy_image = torch.poisson(image * scale) / scale
        return torch.clamp(noisy_image, 0, 1)

    @staticmethod
    def add_speckle_noise(image, mean=0.0, std=0.25):
        """斑点噪声（乘性）"""
        noise = torch.randn_like(image) * std + mean
        noisy_image = image * (1 + noise)
        return torch.clamp(noisy_image, 0, 1)

    # ---------- 前向 ----------
    def __call__(self, tensor):
        if self.use_mixed_noise:
            noise_type = random.choice(config.noise_types)
            noise_param = random.choice(config.noise_params)
        else:
            noise_type = self.noise_type
            noise_param = self.noise_param

        if noise_type == 'gaussian':
            return self.add_gaussian_noise(tensor, mean=0.0, std=max(noise_param, 0.01))
        elif noise_type == 'salt_pepper':
            return self.add_salt_and_pepper_noise(tensor, prob=max(noise_param, 0.01))
        elif noise_type == 'poisson':
            return self.add_poisson_noise(tensor, scale=1.0)
        elif noise_type == 'speckle':
            return self.add_speckle_noise(tensor, mean=0.0, std=max(noise_param, 0.05))
        else:
            return tensor


# ============================================================
# 模型
# ============================================================
class neuronODE(torch.nn.Module):
    def __init__(self):
        super(neuronODE, self).__init__()
        self.gamma = None

    def fresh(self, gamma):
        self.gamma = gamma

    def forward(self, t, p):
        dpdt = -p + torch.pow(torch.sin(config.xi1 * p + torch.cos(config.xi2 * p + self.gamma)), 2)
        return dpdt


class nmODE2(torch.nn.Module):
    def __init__(self):
        super(nmODE2, self).__init__()
        self.neuronODE = neuronODE()
        self.xsize = 15 * 32 * 32
        self.ysize = config.ysize
        self.asize = config.asize

        self.w1 = torch.nn.Linear(self.xsize, self.ysize, True)
        self.w2 = torch.nn.Linear(self.ysize, self.asize, False)
        self.alpha = config.alpha
        self.t = torch.tensor([float(config.t0), float(config.tbar)], device=config.device)

        self.image_processor = ImageProcessor()
        self.batch_norm = torch.nn.BatchNorm1d(self.xsize)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        processed_data = self.image_processor(data)
        batch_size = processed_data.shape[0]
        x = processed_data.view(batch_size, -1)
        x = self.batch_norm(x)
        x = self.dropout(x)

        y0 = torch.zeros(batch_size, self.ysize, device=config.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)
        yt = odeint(self.neuronODE, y0, self.t)[-1]
        at = self.w2(yt)
        return at


# ============================================================
# 数据变换（噪声版：ToTensor 在前，NoiseTransform 在后）
# ============================================================
def create_mixed_noise_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        NoiseTransform(use_mixed_noise=True)
    ])


def create_fixed_noise_transform(noise_type, noise_param):
    return transforms.Compose([
        transforms.ToTensor(),
        NoiseTransform(noise_type=noise_type, noise_param=noise_param, use_mixed_noise=False)
    ])


# ============================================================
# 早停
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_acc, epoch, model, optimizer, path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_acc, epoch, model, optimizer, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(val_acc, epoch, model, optimizer, path)
            self.counter = 0

    def save_checkpoint(self, val_acc, epoch, model, optimizer, path):
        if self.verbose:
            logger.info(f'Validation accuracy improved ({self.best_score:.6f} --> {val_acc:.6f}). Saving model...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': val_acc,
            'best_epoch': epoch
        }, path)


# ============================================================
# 单 epoch 训练 / 评估
# ============================================================
def train_epoch(net, trainloader, criterion, optimizer, epoch_id):
    net.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with tqdm(trainloader, desc=f"Epoch {epoch_id}") as pbar:
        for batch_id, (data, label) in enumerate(pbar):
            label = label.long()
            if data.dim() == 3:
                data = data.unsqueeze(1)
            data = data.view(data.shape[0], 3, 32, 32)
            data = data / 255.0

            optimizer.zero_grad()
            data, label = data.to(config.device), label.to(config.device)

            at = net(data)
            cost = criterion(at, label)
            cost.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            pred = at.argmax(dim=1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)
            total_loss += cost.item() * label.size(0)

            batch_acc = correct / label.size(0)
            pbar.set_postfix({'Loss': f'{cost.item():.4f}', 'Acc': f'{batch_acc:.4f}'})

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(net, testloader, criterion):
    net.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, label in testloader:
            label = label.long()
            if data.dim() == 3:
                data = data.unsqueeze(1)
            data = data.view(data.shape[0], 3, 32, 32)
            data = data / 255.0

            data, label = data.to(config.device), label.to(config.device)
            at = net(data)
            cost = criterion(at, label)

            pred = at.argmax(dim=1)
            correct = (pred == label).sum().item()
            total_correct += correct
            total_samples += label.size(0)
            total_loss += cost.item() * label.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ============================================================
# 单次完整实验（一次 seed）
# ============================================================
def run_single_experiment(seed, args):
    """
    用指定 seed 跑一次完整训练，返回测试指标。
    模型、优化器、数据加载器全部重新初始化，保证独立性。
    """
    set_seed(seed)
    logger.info(f"\n{'='*70}")
    logger.info(f"  Seed = {seed}")
    logger.info(f"{'='*70}")

    # 数据变换（每个 seed 都重新构建，保证数据增强的随机性由 seed 控制）
    if args.no_noise:
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.mixed_noise:
        transform = create_mixed_noise_transform()
    else:
        transform = create_fixed_noise_transform(args.noise_type, args.noise_param)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    # 重新初始化模型
    net = nmODE2().to(config.device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.alpha)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    # 每个 seed 单独的 checkpoint 路径，避免互相覆盖
    checkpoint_path = os.path.join(config.save_path, f'checkpoint_seed{seed}.pth')

    test_acc_history = []
    best_test_acc = 0.0
    best_test_loss = float('inf')
    best_epoch = 0
    actual_epochs = 0

    import time
    start_time = time.time()

    for epoch_id in range(config.epoch):
        train_loss, train_acc = train_epoch(net, trainloader, criterion, optimizer, epoch_id)
        test_loss, test_acc = evaluate(net, testloader, criterion)
        test_acc_history.append(test_acc)

        scheduler.step(test_acc)

        logger.info(f"Epoch: {epoch_id}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch_id
            logger.info(f"  New best accuracy: {test_acc:.4f}")

        early_stopping(test_acc, epoch_id, net, optimizer, checkpoint_path)

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch_id}")
            logger.info(f"Best validation accuracy: {early_stopping.best_score:.4f} "
                        f"at epoch {early_stopping.best_epoch}")
            actual_epochs = epoch_id + 1
            break

        actual_epochs = epoch_id + 1

    total_time = time.time() - start_time

    # 加载最佳模型，做最终测试
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        final_test_loss, final_test_acc = evaluate(net, testloader, criterion)
        logger.info(f"[Seed={seed}] Final test - Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.4f}")
    else:
        final_test_loss, final_test_acc = best_test_loss, best_test_acc

    return {
        'seed': seed,
        'test_accuracy': final_test_acc,
        'test_loss': final_test_loss,
        'best_test_accuracy': best_test_acc,
        'best_epoch': best_epoch,
        'epochs_run': actual_epochs,
        'total_time': total_time,
    }


# ============================================================
# 主程序：5 次重复 + 均值 ± 标准差
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='混合噪声训练 + 5次重复种子 + 标准差')
    parser.add_argument('--mixed_noise', action='store_true', help='使用混合噪声')
    parser.add_argument('--noise_type', type=str, default='salt_pepper',
                        choices=['gaussian', 'salt_pepper', 'poisson', 'speckle'],
                        help='噪声类型')
    parser.add_argument('--noise_param', type=float, default=0.02, help='噪声强度参数')
    parser.add_argument('--patience', type=int, default=30, help='早停耐心值')
    parser.add_argument('--min_delta', type=float, default=0.0001, help='最小改进阈值')
    parser.add_argument('--no_noise', action='store_true', help='不使用噪声')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                        help='重复实验的随机种子列表')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    logger.info(f"使用设备: {config.device}")
    logger.info(f"模型配置:")
    logger.info(f"  - ImageProcessor层数: {config.layer_num * 3}")
    logger.info(f"  - nmODE输入维度: {15 * 32 * 32}")
    logger.info(f"  - nmODE隐藏层维度: {config.ysize}")
    logger.info(f"  - nmODE输出维度: {config.asize}")
    logger.info(f"  - 学习率: {config.alpha}")
    logger.info(f"  - 批量大小: {config.batch_size}")
    logger.info(f"  - 噪声类型: {args.noise_type}, 强度: {args.noise_param}")
    logger.info(f"  - 混合噪声: {args.mixed_noise}, 无噪声: {args.no_noise}")
    logger.info(f"  - 重复种子: {args.seeds}")

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # ========== 对每个 seed 重复实验 ==========
    all_results = []
    for seed in args.seeds:
        result = run_single_experiment(seed, args)
        all_results.append(result)
        logger.info(f"[Seed={seed}] Acc={result['test_accuracy']:.4f}, "
                    f"Loss={result['test_loss']:.4f}, "
                    f"BestEpoch={result['best_epoch']}, "
                    f"Time={result['total_time']:.1f}s")

    # ========== 汇总：均值 ± 标准差 ==========
    print("\n" + "="*70)
    print(f"  Final Summary: {len(args.seeds)} runs, mean ± std")
    print("="*70)
    summary = {}
    for key, name in [
        ('test_accuracy', 'Test Accuracy'),
        ('test_loss', 'Test Loss'),
        ('best_test_accuracy', 'Best Test Accuracy'),
        ('best_epoch', 'Best Epoch'),
        ('total_time', 'Total Time (s)'),
    ]:
        m, s = print_mean_std(all_results, key, name)
        summary[key] = {'mean': m, 'std': s}

    print("\n" + "="*70)
    print("  Summary (copy-paste ready):")
    print("="*70)
    for key, name in [('test_accuracy', 'Test Accuracy'),
                      ('test_loss', 'Test Loss'),
                      ('best_test_accuracy', 'Best Test Accuracy')]:
        print(f"  {name:22s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    # ========== 保存结果 ==========
    out_path = os.path.join(config.save_path, 'multi_seed_summary.json')
    with open(out_path, 'w') as f:
        json.dump({
            'seeds': args.seeds,
            'args': vars(args),
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    logger.info("The program execution is completed.")