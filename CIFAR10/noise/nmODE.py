import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
import numpy as np
import random
import logging
import json
import time
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned


# ============================================================
# 日志系统（同第一个代码）
# ============================================================
def setup_logger():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/training_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

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
# 统一随机种子设置（同第一个代码）
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 均值 ± 标准差工具（同第一个代码）
# ============================================================
def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return float(mean), float(std)


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
    epoch = 10
    batch_size = 256
    num_workers = 0            # ← 保证噪声随机可复现
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger_name = "train-noise2-nmODE"
    save_path = "../nmode/best/models/"

    xsize = 3072               # 3 * 32 * 32
    ysize = 2048
    asize = 10
    alpha = 0.001
    tbar = 2.0
    t0 = 0
    xi1 = 0.49
    xi2 = 0.49
    in_chans = 3
    num_classes = 10

    # 早停
    patience = 30
    min_delta = 0.0


config = Config()
os.makedirs(config.save_path, exist_ok=True)


# ============================================================
# ODE 模型（保持第二个代码原样）
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
        self.xsize = config.xsize
        self.ysize = config.ysize
        self.asize = config.asize

        self.w1 = torch.nn.Linear(config.xsize, config.ysize, True)
        self.w2 = torch.nn.Linear(config.ysize, config.asize, False)

        self.alpha = config.alpha
        self.t = torch.tensor([float(config.t0), float(config.tbar)], device=config.device)

    def forward(self, data):
        batch_size = data.shape[0]
        x = data.to(config.device).view(batch_size, -1)
        y0 = torch.zeros(batch_size, self.ysize, device=config.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)
        yt = odeint(self.neuronODE, y0, self.t)[-1]
        at = self.w2(yt)
        return at

    def save(self, path):
        torch.save({"w1": self.w1.weight, "b": self.w1.bias, "w2": self.w2.weight}, path)

    def load(self, path):
        params = torch.load(path, map_location=config.device)
        self.w1.weight = params["w1"]
        self.w2.weight = params["w2"]
        self.w1.bias = params["b"]


# ============================================================
# 噪声函数（修复原代码 bug）
#   输入: ToTensor 之后的 [0, 1] 张量
#   输出: [0, 1] 张量
# ============================================================
def make_noise_fn(noise_type, noise_param):
    """
    返回作用在 [0,1] 张量上的噪声函数。
    修复：原代码所有 apply_* 都调 add_gaussian_noise；椒盐值域也写错。
    """
    if noise_type == 'gaussian':
        def fn(image):
            noise = torch.randn_like(image) * noise_param
            return torch.clamp(image + noise, 0.0, 1.0)

    elif noise_type == 'salt_pepper':
        def fn(image):
            img_np = image.clone().numpy()
            rnd = np.random.rand(*img_np.shape)
            img_np[rnd < noise_param / 2] = 0.0
            img_np[rnd > 1.0 - noise_param / 2] = 1.0
            return torch.from_numpy(img_np).float()

    elif noise_type == 'poisson':
        def fn(image):
            noisy = torch.poisson(image * noise_param) / noise_param
            return torch.clamp(noisy, 0.0, 1.0)

    elif noise_type == 'speckle':
        def fn(image):
            noise = torch.randn_like(image) * noise_param
            return torch.clamp(image * (1.0 + noise), 0.0, 1.0)

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return fn


def params_to_noise_args(params):
    nt = params['noise_type']
    if nt == 'gaussian':
        return nt, params['noise_std']
    elif nt == 'salt_pepper':
        return nt, params['salt_pepper_prob']
    elif nt == 'poisson':
        return nt, params['poisson_scale']
    elif nt == 'speckle':
        return nt, params['speckle_std']
    else:
        raise ValueError(nt)


# ============================================================
# 单次完整实验（一次 seed）
#   模型/优化器/DataLoader 全部重新初始化
# ============================================================
def run_single_experiment(noise_type, noise_param, seed,
                          epochs=None, verbose=False, trial=None):
    set_seed(seed)
    epochs = epochs if epochs is not None else config.epoch

    if verbose:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"  Seed = {seed}, noise_type={noise_type}, param={noise_param:.5f}")
        logger.info(f"{'=' * 70}")

    # ---- 数据 ----
    # 变换顺序：ToTensor → 噪声 → Normalize（修掉原代码的负值截断 bug）
    noise_fn = make_noise_fn(noise_type, noise_param)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        noise_fn,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers)
    testloader = DataLoader(testset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    # ---- 模型 ----
    net = nmODE2().to(config.device)
    criterion = nn.CrossEntropyLoss(reduction="mean").to(config.device)
    optimizer = torch.optim.SGD(net.parameters(), lr=config.alpha, momentum=0.9)

    best_acc = 0.0
    best_epoch = -1
    last_acc = 0.0
    last_loss = float('inf')
    patience_counter = 0
    epochs_run = 0

    start_time = time.time()

    for epoch_id in range(epochs):
        # ---------------- Train ----------------
        net.train()
        pbar = tqdm(trainloader, desc=f"Seed={seed} Epoch={epoch_id}",
                    disable=not verbose)
        for batch_id, (data, label) in enumerate(pbar):
            label = label.long()
            data = data.view(data.shape[0], 3, 32, 32)

            optimizer.zero_grad()
            data, label = data.to(config.device), label.to(config.device)
            at = net(data)
            cost = criterion(at, label)
            cost.backward()
            optimizer.step()

            if verbose:
                batch_acc = (at.argmax(dim=1) == label).sum().item() / label.shape[0]
                pbar.set_postfix(acc=f"{batch_acc:.4f}")

        # ---------------- Eval ----------------
        net.eval()
        total, correct = 0, 0
        loss_sum = 0.0
        with torch.no_grad():
            for data, label in testloader:
                label = label.long()
                data = data.view(data.shape[0], 3, 32, 32)
                data, label = data.to(config.device), label.to(config.device)
                at = net(data)
                loss = criterion(at, label)
                loss_sum += loss.item() * label.shape[0]
                pred = at.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.shape[0]

        last_acc = correct / total
        last_loss = loss_sum / total
        epochs_run = epoch_id + 1

        if last_acc > best_acc:
            best_acc = last_acc
            best_epoch = epoch_id
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            logger.info(f"Epoch {epoch_id}: Test acc={last_acc:.4f}, loss={last_loss:.4f} | "
                        f"best={best_acc:.4f}@{best_epoch} | patience={patience_counter}/{config.patience}")

        # Optuna pruner 中间上报
        if trial is not None:
            trial.report(last_acc, step=epoch_id)
            if trial.should_prune():
                raise TrialPruned()

        # 早停
        if patience_counter >= config.patience:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch_id}")
            break

    total_time = time.time() - start_time

    return {
        'seed': seed,
        'noise_type': noise_type,
        'noise_param': float(noise_param),
        'accuracy': float(last_acc),
        'test_loss': float(last_loss),
        'best_accuracy': float(best_acc),
        'best_epoch': int(best_epoch),
        'epochs_run': int(epochs_run),
        'total_time': float(total_time),
    }


# ============================================================
# Optuna 目标函数（同第一个代码结构）
# ============================================================
def make_objective(search_epochs, trial_seed=42):
    def objective(trial):
        try:
            noise_type = trial.suggest_categorical(
                'noise_type', ['gaussian', 'salt_pepper', 'poisson', 'speckle'])

            if noise_type == 'gaussian':
                trial.suggest_float('noise_std', 0.01, 0.3)
            elif noise_type == 'salt_pepper':
                trial.suggest_float('salt_pepper_prob', 0.01, 0.1)
            elif noise_type == 'poisson':
                trial.suggest_float('poisson_scale', 0.1, 5.0)
            elif noise_type == 'speckle':
                trial.suggest_float('speckle_std', 0.01, 0.5)

            nt, np_ = params_to_noise_args(trial.params)
            logger.info(f"Trial {trial.number}: noise_type={nt}, param={np_:.5f}")

            res = run_single_experiment(
                noise_type=nt,
                noise_param=np_,
                seed=trial_seed,      # 所有 trial 用同 seed，让差异来自噪声参数
                epochs=search_epochs,
                verbose=False,
                trial=trial,
            )
            logger.info(f"Trial {trial.number} - best acc: {res['best_accuracy']:.4f}")
            return res['best_accuracy']

        except TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            logger.exception("Detailed error:")
            return 0.0

    return objective


# ============================================================
# 主程序：Optuna 搜索 → 5 次重复 → 均值 ± 标准差
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='nmODE2 + Optuna noise search + 5 runs')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Optuna 搜索 trial 数')
    parser.add_argument('--search_epochs', type=int, default=10,
                        help='搜索阶段每个 trial 训练多少 epoch（建议调小加速）')
    parser.add_argument('--final_epochs', type=int, default=None,
                        help='最终 5 次重复训练多少 epoch；默认 config.epoch')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                        help='重复实验的随机种子列表')
    parser.add_argument('--verbose', action='store_true',
                        help='是否显示每个 epoch 的进度条')
    args = parser.parse_args()

    if args.final_epochs is None:
        args.final_epochs = config.epoch

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    logger.info(f"使用设备: {config.device}")
    logger.info(f"重复种子: {args.seeds}")

    # ========== Step 1: Optuna 搜索 ==========
    logger.info("=" * 70)
    logger.info(f"Start Optuna optimization: n_trials={args.n_trials}, "
                f"search_epochs={args.search_epochs}")
    logger.info("=" * 70)

    set_seed(42)
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f"nmODE2_noise_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    def log_study(study, trial):
        logger.info(f"Trial {trial.number} finished - Value: {trial.value}")

    study.optimize(make_objective(args.search_epochs, trial_seed=42),
                   n_trials=args.n_trials,
                   callbacks=[log_study])

    logger.info("=" * 70)
    logger.info("Optuna 搜索完成")
    logger.info(f"  Best trial : {study.best_trial.number}")
    logger.info(f"  Best value : {study.best_value:.4f}")
    logger.info(f"  Best params: {study.best_params}")
    logger.info("=" * 70)

    optuna_result_path = os.path.join(config.save_path, 'optuna_best_params.json')
    with open(optuna_result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_value': float(study.best_value),
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Optuna 最佳结果已保存到 {optuna_result_path}")

    # ========== Step 2: 用最佳参数在 N 个 seed 上重复实验 ==========
    best_nt, best_np_ = params_to_noise_args(study.best_params)

    logger.info("=" * 70)
    logger.info(f"开始 {len(args.seeds)} 次重复实验 "
                f"(noise_type={best_nt}, param={best_np_:.5f})")
    logger.info("=" * 70)

    all_results = []
    for seed in args.seeds:
        result = run_single_experiment(
            noise_type=best_nt,
            noise_param=best_np_,
            seed=seed,
            epochs=args.final_epochs,
            verbose=args.verbose,
        )
        all_results.append(result)
        logger.info(f"[Seed={seed}] Acc={result['accuracy']:.4f}, "
                    f"Loss={result['test_loss']:.4f}, "
                    f"BestAcc={result['best_accuracy']:.4f}@{result['best_epoch']}, "
                    f"Time={result['total_time']:.1f}s")

    # ========== Step 3: 汇总 均值 ± 标准差 ==========
    print("\n" + "=" * 70)
    print(f"  Final Summary: {len(args.seeds)} runs, mean ± std")
    print("=" * 70)
    summary = {}
    for key, name in [
        ('accuracy', 'Test Accuracy'),
        ('test_loss', 'Test Loss'),
        ('best_accuracy', 'Best Test Accuracy'),
        ('best_epoch', 'Best Epoch'),
        ('epochs_run', 'Epochs Run'),
        ('total_time', 'Total Time (s)'),
    ]:
        m, s = print_mean_std(all_results, key, name)
        summary[key] = {'mean': m, 'std': s}

    print("\n" + "=" * 70)
    print("  Summary (copy-paste ready):")
    print("=" * 70)
    for key, name in [
        ('accuracy', 'Test Accuracy'),
        ('test_loss', 'Test Loss'),
        ('best_accuracy', 'Best Test Accuracy'),
    ]:
        print(f"  {name:22s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    # ========== Step 4: 保存结果 ==========
    out_path = os.path.join(config.save_path, 'multi_seed_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': study.best_params,
            'optuna_best_value': float(study.best_value),
            'seeds': args.seeds,
            'search_epochs': args.search_epochs,
            'final_epochs': args.final_epochs,
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")
    logger.info("The program execution is completed.")


if __name__ == "__main__":
    main()

# # 只搜索高斯噪声
# python train.py --noise_type gaussian --n_trials 20
#
# # 只搜索椒盐噪声
# python train.py --noise_type salt_pepper --n_trials 20
#
# # 两种噪声都搜一遍，自动选最优（默认）
# python train.py --noise_type both --n_trials 20
#
# # 加 --verbose 看训练进度条
# python train.py --noise_type both --verbose