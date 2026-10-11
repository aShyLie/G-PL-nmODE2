import numpy as np
import os
import torch
from torchdiffeq import odeint
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from PIL import ImageFilter, Image
import torch.nn as nn
import cv2
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import optuna
from optuna.pruners import MedianPruner
import random
import logging
import json
from datetime import datetime

# ======================================================================
#   Final Summary: 5 runs, mean ± std
# ======================================================================
# Accuracy: 0.3514 ± 0.0082
# AUC-ROC: 0.7834 ± 0.0028
# Log Loss: 2.1688 ± 0.0018
# Precision (weighted): 0.3782 ± 0.0044
# Recall (weighted): 0.3514 ± 0.0082
# F1 (weighted): 0.3463 ± 0.0128
#
# ======================================================================
#   Summary (copy-paste ready):
# ======================================================================
#   Accuracy    : 0.3514 ± 0.0082
#   AUC-ROC     : 0.7834 ± 0.0028
#   Log Loss    : 2.1688 ± 0.0018
#   Precision   : 0.3782 ± 0.0044
#   Recall      : 0.3514 ± 0.0082
#   F1          : 0.3463 ± 0.0128
#
# Saved to ./models/cifar10_best_params_and_summary.json
# 2026-09-19 08:47:36,481 - INFO - Accuracy: 0.3514 ± 0.0082
# 2026-09-19 08:47:36,482 - INFO - AUC-ROC: 0.7834 ± 0.0028
# 2026-09-19 08:47:36,482 - INFO - Log Loss: 2.1688 ± 0.0018
# 2026-09-19 08:47:36,483 - INFO - Precision (weighted): 0.3782 ± 0.0044
# 2026-09-19 08:47:36,483 - INFO - Recall (weighted): 0.3514 ± 0.0082
# 2026-09-19 08:47:36,483 - INFO - F1 (weighted): 0.3463 ± 0.0128
# 2026-09-19 08:47:36,492 - INFO - The program execution is completed.

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
# 模糊变换
# ============================================================
class BlurTransform(object):
    def __init__(self, blur_type='gaussian', blur_radius=5):
        self.blur_type = blur_type
        self.blur_radius = blur_radius

    def __call__(self, img):
        img_array = np.array(img)
        if self.blur_type == 'gaussian':
            blurred = cv2.GaussianBlur(img_array, (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1),
                                       self.blur_radius)
        elif self.blur_type == 'average':
            blurred = cv2.blur(img_array, (self.blur_radius * 2 + 1, self.blur_radius * 2 + 1))
        elif self.blur_type == 'median':
            blurred = cv2.medianBlur(img_array, self.blur_radius * 2 + 1)
        elif self.blur_type == 'motion':
            size = self.blur_radius * 2 + 1
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size)
            kernel = kernel / size
            blurred = cv2.filter2D(img_array, -1, kernel)
        elif self.blur_type == 'bilateral':
            blurred = cv2.bilateralFilter(img_array, 9, self.blur_radius * 10, self.blur_radius * 10)
        else:
            blurred = img_array
        return Image.fromarray(blurred)


# ============================================================
# 配置
# ============================================================
class Config:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 64
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
        self.layers_num = 15
        self.verbose = 2
        self.weight_decay = 1e-4
        self.num_workers = 0

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        logger.info(f"device: {self.device}")

        self.logger_name = "train"
        self.best_epoch = 0
        self.best_acc = 0
        self.save_path = "./models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.xsize = 3 * 32 * 32
        self.ysize = 64
        self.asize = 10
        self.alpha = 0.00001
        self.tbar = 0.5
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.in_chans = 3

        self.blur_type = 'average'
        self.blur_radius = 5
        self.blur_types = ['gaussian', 'average', 'median', 'motion', 'bilateral']
        self.blur_radii = [3, 5, 7, 9]


config = Config()


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
        self.xsize = config.xsize
        self.ysize = config.ysize
        self.asize = config.asize

        self.w1 = torch.nn.Linear(config.xsize, config.ysize, True).to(config.device)
        self.w2 = torch.nn.Linear(config.ysize, config.asize, False).to(config.device)

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


# ============================================================
# ImageProcessor
# ============================================================
class ImageProcessor:
    def __init__(self, config):
        self.layer_num = config.layers_num
        self.config = config
        self.model = nmODE2()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.alpha,
                              weight_decay=config.weight_decay)

        self.preprocessed_data_path = './processed_data'
        os.makedirs(self.preprocessed_data_path, exist_ok=True)
        self.model.to(config.device)
        logger.info(f"Model initialized and moved to {config.device}")
        self.blur_transform = BlurTransform(blur_type=config.blur_type,
                                            blur_radius=config.blur_radius)

    def load_preprocess_image(self, image, blur_type, blur_radius):
        image = Image.fromarray(image.astype(np.uint8))
        if image.size != (32, 32):
            image = image.resize((32, 32))

        if blur_radius > 0:
            blur_transform = BlurTransform(blur_type=blur_type, blur_radius=blur_radius)
            image = blur_transform(image)

        image = np.array(image)
        image = image.reshape((32, 32, 3))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = (image - mean) / std
        return image

    def process_and_save_images(self, images, file_name, alpha, blur_type, blur_radius):
        """缓存文件名包含 blur_type/blur_radius，避免不同参数复用旧缓存"""
        cache_name = f"{file_name}_bt{blur_type}_br{blur_radius}.pt"
        processed_images_file = os.path.join(self.preprocessed_data_path, cache_name)
        if os.path.exists(processed_images_file):
            print(f"{cache_name} already exists. Loading the data.")
            return torch.load(processed_images_file, map_location=config.device)

        print(f"Processing {cache_name} and saving.")
        processed_images = np.array([
            self.load_preprocess_image(img, blur_type, blur_radius) for img in images
        ])
        processed_images = torch.from_numpy(processed_images).float()
        processed_images = processed_images.permute(0, 3, 1, 2)
        torch.save(processed_images, processed_images_file)
        return processed_images.to(config.device)

    def load_cifar10_data(self, alpha, blur_type, blur_radius):
        logger.info("Load the CIFAR-10 dataset...")
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

        train_images = train_dataset.data
        train_labels = train_dataset.targets
        test_images = test_dataset.data
        test_labels = test_dataset.targets

        processed_train_images = self.process_and_save_images(
            train_images, 'processed_train_images.pt', alpha, blur_type, blur_radius)
        processed_test_images = self.process_and_save_images(
            test_images, 'processed_test_images.pt', alpha, blur_type, blur_radius)

        transformed_train_images = processed_train_images
        transformed_test_images = processed_test_images

        train_labels = torch.tensor(train_labels, dtype=torch.long, device=config.device)
        test_labels = torch.tensor(test_labels, dtype=torch.long, device=config.device)

        assert not torch.isnan(transformed_train_images).any()
        assert not torch.isnan(transformed_test_images).any()
        assert not torch.isinf(transformed_train_images).any()
        assert not torch.isinf(transformed_test_images).any()

        logger.info(f"Data shapes - Train: {transformed_train_images.shape}, "
                    f"Test: {transformed_test_images.shape}")
        return transformed_train_images, train_labels, transformed_test_images, test_labels

    def train_model(self, transformed_train_images, train_labels,
                    transformed_test_images, test_labels):
        train_dataset = TensorDataset(transformed_train_images, train_labels)
        test_dataset = TensorDataset(transformed_test_images, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        best_val_accuracy = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_f1_score = 0.0
        best_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None

        patience = 5
        min_delta = 0.001
        patience_count = 0

        training_history = {'epoch': [], 'train_loss': [], 'train_acc': [],
                            'val_loss': [], 'val_acc': []}

        start_time = time.time()
        logger.info(f"Start training. Total epochs: {self.config.epochs}, "
                    f"Batch size: {self.config.batch_size}")

        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(config.device), target.to(config.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct_train += pred.eq(target.view_as(pred)).sum().item()
                total_train += target.size(0)

            avg_train_loss = total_train_loss / total_train
            accuracy_train = 100.0 * correct_train / total_train

            # 验证
            self.model.eval()
            all_targets, all_predictions = [], []
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(config.device), target.to(config.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    total_val_loss += loss.item() * data.size(0)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_val += pred.eq(target.view_as(pred)).sum().item()
                    total_val += target.size(0)
                    all_targets.extend(target.cpu().numpy())
                    all_predictions.extend(pred.cpu().numpy().flatten())

            avg_val_loss = total_val_loss / total_val
            accuracy_val = 100.0 * correct_val / total_val
            precision, recall, f1_score, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted', zero_division=0)

            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_acc'].append(accuracy_train)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_acc'].append(accuracy_val)

            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - "
                        f"Train Loss: {avg_train_loss:.4f} - Train Acc: {accuracy_train:.2f}% - "
                        f"Val Loss: {avg_val_loss:.4f} - Val Acc: {accuracy_val:.2f}% - "
                        f"P: {precision:.4f} - R: {recall:.4f} - F1: {f1_score:.4f} - "
                        f"Time: {time.time() - epoch_start_time:.2f}s")

            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_accuracy = accuracy_val
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score
                best_epoch = epoch + 1
                best_model_state = self.model.state_dict()
                patience_count = 0
                logger.info(f"New best val loss: {best_val_loss:.4f}, acc: {best_val_accuracy:.2f}%")
            else:
                patience_count += 1

            if accuracy_val > best_val_accuracy:
                best_val_accuracy = accuracy_val
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score
                logger.info(f"New best val acc: {best_val_accuracy:.2f}%")

            if patience_count >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        model_path = os.path.join(config.save_path, f"best_model_{best_epoch}.pth")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'val_accuracy': best_val_accuracy,
            'training_history': training_history
        }, model_path)
        logger.info(f"The best model is saved in: {model_path}")

        total_time = time.time() - start_time
        logger.info(f'Training completed - Total time: {total_time:.2f}s - Best epoch: {best_epoch} - '
                    f'Best val loss: {best_val_loss:.4f} - Best val acc: {best_val_accuracy:.2f}% - '
                    f'Best precision: {best_precision:.4f} - Best recall: {best_recall:.4f} - '
                    f'Best F1: {best_f1_score:.4f}')

        self.model.load_state_dict(best_model_state)
        return best_val_accuracy, best_model_state, training_history

    def evaluate_model(self, transformed_test_images, test_labels):
        self.model.eval()
        all_probs, all_targets = [], []
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(transformed_test_images, test_labels),
                                     batch_size=self.config.batch_size)
            for data, target in test_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
        predictions = np.argmax(all_probs, axis=1)
        accuracy = accuracy_score(all_targets, predictions)
        try:
            auc_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
        except:
            auc_roc = 0.5
        logloss = log_loss(all_targets, all_probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets, predictions)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        print(f'准确率: {accuracy:.4f}')
        print(f'AUC-ROC: {auc_roc:.4f}')
        print(f'对数损失: {logloss:.4f}')

        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'logloss': logloss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_accuracies': class_accuracies,
        }


# ============================================================
# Optuna 目标函数
# ============================================================
def objective(trial):
    try:
        # 固定 seed，保证 Optuna 搜索阶段可复现
        set_seed(42)

        cfg = Config()
        blur_type = trial.suggest_categorical('blur_type', cfg.blur_types)
        blur_radius = trial.suggest_int('blur_radius', 3, 9, step=2)
        alpha = trial.suggest_float('alpha', 0.0, 1.0)

        cfg.blur_type = blur_type
        cfg.blur_radius = blur_radius

        logger.info(f"Trial {trial.number}: blur_type={blur_type}, "
                    f"blur_radius={blur_radius}, alpha={alpha}")

        processor = ImageProcessor(cfg)
        transformed_train_images, train_labels, transformed_test_images, test_labels = \
            processor.load_cifar10_data(alpha, blur_type, blur_radius)

        best_val_accuracy, _, _ = processor.train_model(
            transformed_train_images, train_labels,
            transformed_test_images, test_labels)

        eval_res = processor.evaluate_model(transformed_test_images, test_labels)
        composite_score = (eval_res['accuracy'] * 0.4 +
                           eval_res['auc_roc'] * 0.3 +
                           (1 - eval_res['logloss']) * 0.3)
        logger.info(f"Trial {trial.number} - composite score: {composite_score:.4f}")
        return composite_score
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 0.0


# ============================================================
# 单次完整实验（用于 5 次重复种子）
# ============================================================
def run_single_experiment(seed, best_params):
    """用最优参数 + 指定 seed 训练一次，返回全部指标"""
    set_seed(seed)

    cfg = Config()
    cfg.blur_type = best_params['blur_type']
    cfg.blur_radius = best_params['blur_radius']
    alpha = best_params['alpha']

    logger.info(f"\n{'='*70}")
    logger.info(f"  Seed = {seed}")
    logger.info(f"  blur_type={cfg.blur_type}, blur_radius={cfg.blur_radius}, alpha={alpha}")
    logger.info(f"{'='*70}")

    processor = ImageProcessor(cfg)
    transformed_train_images, train_labels, transformed_test_images, test_labels = \
        processor.load_cifar10_data(alpha, cfg.blur_type, cfg.blur_radius)

    processor.train_model(transformed_train_images, train_labels,
                          transformed_test_images, test_labels)

    eval_res = processor.evaluate_model(transformed_test_images, test_labels)

    return {
        'seed': seed,
        'accuracy': eval_res['accuracy'],
        'auc_roc': eval_res['auc_roc'],
        'logloss': eval_res['logloss'],
        'precision': eval_res['precision'],
        'recall': eval_res['recall'],
        'f1': eval_res['f1'],
    }


# ============================================================
# 主程序：Optuna 搜索 + 5 次重复 + 报告均值 ± 标准差
# ============================================================
def main():
    # ========== Step 1: Optuna 搜索最佳超参数 ==========
    logger.info("Start Optuna optimization...")
    set_seed(42)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler,
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5))

    def log_study(study, trial):
        logger.info(f"Trial {trial.number} finished - Value: {trial.value}")

    study.optimize(objective, n_trials=10, callbacks=[log_study])

    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_value}")
    logger.info(f"  Params: {study.best_params}")

    best_params = study.best_params

    # ========== Step 2: 用最优参数在 5 个 seed 上重复实验 ==========
    seeds = [42, 43, 44, 45, 46]
    logger.info(f"\n===== 用最优参数在 {len(seeds)} 个 seed 上重复实验 =====")
    logger.info(f"Best params: {best_params}")

    all_results = []
    for seed in seeds:
        result = run_single_experiment(seed, best_params)
        all_results.append(result)
        print(f"[Seed={seed}] Acc={result['accuracy']:.4f}, "
              f"AUC={result['auc_roc']:.4f}, LogLoss={result['logloss']:.4f}, "
              f"F1={result['f1']:.4f}")

    # ========== Step 3: 汇总均值 ± 标准差 ==========
    print("\n" + "="*70)
    print(f"  Final Summary: {len(seeds)} runs, mean ± std")
    print("="*70)
    summary = {}
    for key, name in [
        ('accuracy', 'Accuracy'),
        ('auc_roc', 'AUC-ROC'),
        ('logloss', 'Log Loss'),
        ('precision', 'Precision (weighted)'),
        ('recall', 'Recall (weighted)'),
        ('f1', 'F1 (weighted)'),
    ]:
        m, s = print_mean_std(all_results, key, name)
        summary[key] = {'mean': m, 'std': s}

    print("\n" + "="*70)
    print("  Summary (copy-paste ready):")
    print("="*70)
    for key, name in [('accuracy', 'Accuracy'), ('auc_roc', 'AUC-ROC'),
                      ('logloss', 'Log Loss'), ('precision', 'Precision'),
                      ('recall', 'Recall'), ('f1', 'F1')]:
        print(f"  {name:12s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    # ========== Step 4: 保存结果 ==========
    out_path = os.path.join(config.save_path, 'cifar10_best_params_and_summary.json')
    with open(out_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_value': study.best_value,
            'summary': summary,
            'all_results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    logger.info("The program execution is completed.")


if __name__ == "__main__":
    main()