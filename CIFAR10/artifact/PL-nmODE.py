import os
import argparse
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned

from torchdiffeq import odeint
from tqdm import tqdm
from loguru import logger
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter


# ============================================================
# Config
# ============================================================
class Config:
    epoch = 1000
    batch_size = 256
    num_workers = 8
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger_name = "train"
    best_epoch = 0
    best_acc = 0
    save_path = "./models/"

    layer_num = 5
    xsize = 5 * 32 * 32 * 3
    ysize = 1024
    fsize = 1024
    asize = 10
    alpha = 0.0001
    tbar = 2.0
    t0 = 0
    xi1 = 0.49
    xi2 = 0.49
    in_chans = 3
    num_classes = 10

    # 伪影参数（默认值，会被 Optuna 覆盖）
    noise_std = 0.1
    blur_radius = 5


config = Config()
os.makedirs(config.save_path, exist_ok=True)


# ============================================================
# 种子工具
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ImageProcessor（原逻辑不动）
# ============================================================
class ImageProcessor(torch.nn.Module):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.layer_num = config.layer_num

    def forward(self, data):
        device = data.device
        batch_size, channels, height, width = data.shape
        transformed_images = torch.zeros(
            size=(batch_size, self.layer_num, height, width, channels),
            dtype=torch.float32, device=device
        )

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
                (0 <= pixel) & (pixel <= 63.75), s1, transformed_images[:, 0, :, :, c])
            transformed_images[:, 1, :, :, c] = torch.where(
                (0 <= pixel) & (pixel <= 63.75), s2, transformed_images[:, 1, :, :, c])
            transformed_images[:, 1, :, :, c] = torch.where(
                (64 <= pixel) & (pixel <= 127.5), s3, transformed_images[:, 1, :, :, c])
            transformed_images[:, 2, :, :, c] = torch.where(
                (64 <= pixel) & (pixel <= 127.5), s4, transformed_images[:, 2, :, :, c])
            transformed_images[:, 2, :, :, c] = torch.where(
                (128 <= pixel) & (pixel <= 191.25), s5, transformed_images[:, 2, :, :, c])
            transformed_images[:, 3, :, :, c] = torch.where(
                (128 <= pixel) & (pixel <= 191.25), s6, transformed_images[:, 3, :, :, c])
            transformed_images[:, 3, :, :, c] = torch.where(
                (192 <= pixel) & (pixel <= 255), s7, transformed_images[:, 3, :, :, c])
            transformed_images[:, 4, :, :, c] = torch.where(
                (192 <= pixel) & (pixel <= 255), s8, transformed_images[:, 4, :, :, c])

        transformed_images = transformed_images.permute(0, 4, 1, 2, 3)
        transformed_images = transformed_images.reshape(batch_size, -1, height, width)
        return transformed_images


# ============================================================
# ODE 模型
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
        self.fsize = config.fsize
        self.asize = config.asize
        self.w1 = torch.nn.Linear(config.xsize, config.ysize, True)
        self.w2 = torch.nn.Linear(config.fsize, config.asize, False)
        self.alpha = config.alpha
        self.t = torch.tensor([float(config.t0), float(config.tbar)], device=config.device)

    def forward(self, data):
        batch_size = data.shape[0]
        x = data.to(config.device)
        y0 = torch.zeros(batch_size, self.ysize, device=config.device)
        gamma = self.w1(x.view(x.size(0), -1))
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
# 伪影参数化的 transform（原逻辑，加参数）
# ============================================================
def make_transform(noise_std, blur_radius):
    """
    返回一个与 ToTensor 串起来的 transforms.Compose。
    - noise_std：高斯噪声标准差
    - blur_radius：高斯模糊 kernel_size（必须是正奇数）
    """
    # 保证 blur_radius 是奇数
    blur_radius = int(blur_radius)
    if blur_radius % 2 == 0:
        blur_radius += 1
    blur_radius = max(1, blur_radius)

    def apply_artifacts(image):
        noise = torch.randn_like(image) * noise_std
        blurred = transforms.functional.gaussian_blur(image, blur_radius)
        artifacted = (image + noise) / 2 + blurred / 2
        return torch.clamp(artifacted, 0, 1)

    return transforms.Compose([
        transforms.ToTensor(),
        apply_artifacts,
    ])


# ============================================================
# 单次完整训练 + 评估
# ============================================================
def train_and_evaluate(noise_std, blur_radius, seed,
                       epochs=None, verbose=False, trial=None):
    """
    给定伪影参数 + 随机种子，训练一次 nmODE2 并返回 best_acc。
    """
    epochs = epochs if epochs is not None else config.epoch
    set_seed(seed)

    transform = make_transform(noise_std, blur_radius)

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=config.batch_size,
                             shuffle=True, num_workers=config.num_workers)
    testloader = DataLoader(testset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)

    device = config.device
    net = nmODE2().to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.alpha)

    # ImageProcessor 只 new 一次（原代码每个 batch 都 new 一次，很浪费）
    process = ImageProcessor()

    best_acc = 0.0
    best_epoch = -1
    last_acc = 0.0

    for epoch_id in range(epochs):
        # ---------------- Train ----------------
        net.train()
        pbar = tqdm(trainloader, desc=f"Epoch {epoch_id}", disable=not verbose)
        for batch_id, (data, label) in enumerate(pbar):
            label = label.long()
            data = data.view(data.shape[0], 3, 32, 32)
            data = data * 255.0
            data = process.forward(data)

            optimizer.zero_grad()
            data, label = data.to(device), label.to(device)
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
        with torch.no_grad():
            for data, label in testloader:
                label = label.long()
                data = data.view(data.shape[0], 3, 32, 32)
                data = data * 255.0
                data = process.forward(data).to(device)
                a_pred = net(data).cpu().argmax(dim=1)
                correct += (a_pred == label).sum().item()
                total += label.shape[0]

        last_acc = correct / total
        if last_acc > best_acc:
            best_acc = last_acc
            best_epoch = epoch_id

        if verbose:
            logger.info(f"Epoch {epoch_id} Test acc={last_acc:.4f} | Best={best_acc:.4f}@{best_epoch}")

        # Optuna 中间上报（用于 pruner）
        if trial is not None:
            trial.report(last_acc, step=epoch_id)
            if trial.should_prune():
                raise TrialPruned()

    return {
        'best_acc': float(best_acc),
        'best_epoch': int(best_epoch),
        'last_acc': float(last_acc),
    }


# ============================================================
# Optuna objective：搜索 noise_std + blur_radius
# ============================================================
def make_objective(search_epochs, trial_seed=42):
    """
    - 所有 trial 用相同的 trial_seed → 差异来自伪影参数，对比公平
    - search_epochs 一般比最终 epoch 小，加快搜索
    """
    def objective(trial):
        try:
            noise_std = trial.suggest_float('noise_std', 0.01, 0.3)
            # gaussian_blur 要求 kernel_size 是正奇数 → 用 step=2 保证是奇数
            blur_radius = trial.suggest_int('blur_radius', 3, 15, step=2)

            logger.info(f"[Trial {trial.number}] noise_std={noise_std:.4f}, blur_radius={blur_radius}")

            res = train_and_evaluate(
                noise_std=noise_std,
                blur_radius=blur_radius,
                seed=trial_seed,
                epochs=search_epochs,
                verbose=False,
                trial=trial,
            )
            logger.info(f"[Trial {trial.number}] best_acc={res['best_acc']:.4f}")
            return res['best_acc']

        except TrialPruned:
            raise
        except Exception as e:
            logger.exception(f"[Trial {trial.number}] failed: {e}")
            return 0.0

    return objective


# ============================================================
# 5 次重复 + 报告
# ============================================================
def run_repeats(best_params, n_repeats, base_seed, final_epochs, verbose=False):
    noise_std = best_params['noise_std']
    blur_radius = best_params['blur_radius']

    logger.info("=" * 70)
    logger.info(f"开始 {n_repeats} 次重复实验, "
                f"noise_std={noise_std:.4f}, blur_radius={blur_radius}")
    logger.info("=" * 70)

    results = []
    for i in range(n_repeats):
        seed = base_seed + i
        logger.info(f"--- Repeat {i + 1}/{n_repeats} (seed={seed}) ---")
        try:
            res = train_and_evaluate(
                noise_std=noise_std,
                blur_radius=blur_radius,
                seed=seed,
                epochs=final_epochs,
                verbose=verbose,
            )
            res['seed'] = seed
            results.append(res)
            logger.info(f"[Repeat {i + 1}] seed={seed} "
                        f"best_acc={res['best_acc']:.4f} "
                        f"(best_epoch={res['best_epoch']}, last_acc={res['last_acc']:.4f})")
        except Exception as e:
            logger.exception(f"[Repeat {i + 1}] failed: {e}")

    return results


def report_mean_std(results, metric_key='best_acc'):
    if not results:
        logger.warning("没有成功的重复实验，无法汇总。")
        return {}

    vals = np.array([r[metric_key] for r in results], dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    logger.info("=" * 70)
    logger.info(f"最终结果 ({len(vals)} 次重复) —— {metric_key}")
    logger.info(f"  各次: {np.round(vals, 4).tolist()}")
    logger.info(f"  Mean ± Std:  {mean:.4f} ± {std:.4f}")
    logger.info("=" * 70)

    # 控制台再打印一遍
    print(f"\n{'=' * 70}")
    print(f"最终结果 ({len(vals)} 次重复) —— {metric_key}")
    print(f"  各次: {np.round(vals, 4).tolist()}")
    print(f"  Mean ± Std:  {mean:.4f} ± {std:.4f}")
    print(f"{'=' * 70}\n")

    return {
        'metric': metric_key,
        'values': vals.tolist(),
        'mean': mean,
        'std': std,
        'n_repeats': int(len(vals)),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='nmODE2 + Optuna artifact search')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Optuna 搜索 trial 数')
    parser.add_argument('--n_repeats', type=int, default=5,
                        help='用最佳参数重复训练的轮数')
    parser.add_argument('--base_seed', type=int, default=42,
                        help='基础随机种子；重复实验用 base_seed, base_seed+1, ...')
    parser.add_argument('--search_epochs', type=int, default=50,
                        help='Optuna 搜索阶段每个 trial 训练多少 epoch（建议调小加速）')
    parser.add_argument('--final_epochs', type=int, default=None,
                        help='最终 5 次重复训练多少 epoch；默认 config.epoch')
    parser.add_argument('--verbose', action='store_true',
                        help='是否显示每个 epoch 的进度条与日志')
    args = parser.parse_args()

    if args.final_epochs is None:
        args.final_epochs = config.epoch

    # ---- 日志 ----
    logger.remove()
    log_filename = f"{config.logger_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_filename, level='INFO')
    logger.add(lambda msg: print(msg, end=''), level='INFO')
    logger.info(f"日志写入 {log_filename}")

    # ---------- 1. Optuna 搜索 ----------
    logger.info("=" * 70)
    logger.info(f"开始 Optuna 搜索: n_trials={args.n_trials}, "
                f"search_epochs={args.search_epochs}")
    logger.info("=" * 70)

    sampler = TPESampler(seed=args.base_seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name=f"nmODE2_artifact_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    def log_callback(study, trial):
        logger.info(f"[Optuna] Trial {trial.number} finished - value={trial.value}")

    study.optimize(
        make_objective(args.search_epochs, trial_seed=args.base_seed),
        n_trials=args.n_trials,
        callbacks=[log_callback],
    )

    logger.info("=" * 70)
    logger.info(f"Optuna 搜索完成. Best trial={study.best_trial.number}, "
                f"value={study.best_value:.4f}, params={study.best_params}")
    logger.info("=" * 70)

    optuna_path = os.path.join(config.save_path, 'optuna_best_params.json')
    with open(optuna_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_value': float(study.best_value),
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Optuna 结果已保存到 {optuna_path}")

    # ---------- 2. 5 次重复 ----------
    repeat_results = run_repeats(
        best_params=study.best_params,
        n_repeats=args.n_repeats,
        base_seed=args.base_seed,
        final_epochs=args.final_epochs,
        verbose=args.verbose,
    )

    # ---------- 3. 汇总 均值±标准差 ----------
    summary = report_mean_std(repeat_results, metric_key='best_acc')

    summary_path = os.path.join(config.save_path, 'final_repeat_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': study.best_params,
            'optuna_best_value': float(study.best_value),
            'search_epochs': args.search_epochs,
            'final_epochs': args.final_epochs,
            'per_run': repeat_results,
            'summary': summary,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"重复实验汇总已保存到 {summary_path}")

    if summary:
        logger.info(f"[FINAL] best_acc = {summary['mean']:.4f} ± {summary['std']:.4f} "
                    f"(n={summary['n_repeats']})")


if __name__ == '__main__':
    main()