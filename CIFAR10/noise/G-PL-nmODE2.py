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
from optuna.exceptions import TrialPruned
import random
import logging
import json
from datetime import datetime

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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
        self.weight_decay = 0.001

        self.num_workers = 2
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.logger_name = "train"
        self.best_epoch = 0
        self.best_acc = 0
        self.save_path = "./models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.xsize = 15360
        self.ysize = 2048 * 2
        self.asize = 10
        self.alpha = 0.00005
        self.tbar = 2.0
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.in_chans = 3

        # 噪音参数配置
        self.noise_type = 'gaussian'
        self.noise_std = 0.1
        self.salt_pepper_prob = 0.02
        self.speckle_std = 0.25
        self.poisson_scale = 1.0

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
    def __init__(self, input_channels=15, height=32, width=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.xsize = 64 * 16 * 16
        self.ysize = config.ysize
        self.asize = config.asize
        self.w1 = nn.Linear(self.xsize, self.ysize, True)
        self.w2 = nn.Linear(config.ysize, config.asize, False)
        self.alpha = config.alpha
        self.t = torch.tensor([float(config.t0), float(config.tbar)], device=config.device)
        self.neuronODE = neuronODE().to(config.device)

    def forward(self, data):
        batch_size = data.shape[0]
        x = self.conv(data)
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
# 图像处理
# ============================================================
class ImageProcessor:
    def __init__(self, config, seed=None):
        self.layer_num = config.layers_num
        self.config = config
        self.seed = seed
        self.model = nmODE2()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.alpha, weight_decay=config.weight_decay)

        self.preprocessed_data_path = './processed_data'
        self.poisson_scale = getattr(config, 'poisson_scale', 1.0)
        os.makedirs(self.preprocessed_data_path, exist_ok=True)
        self.model.to(config.device)
        logger.info(f"模型初始化完成, 设备: {config.device}")

    # ---- 噪音生成 ----
    def add_gaussian_noise(self, image, mean=0, std=0.1):
        noise = torch.randn_like(image) * std + mean
        return torch.clamp(image + noise, 0, 1)

    def add_salt_and_pepper_noise(self, image, prob=0.02):
        noisy = np.copy(image.numpy())
        noisy[np.random.rand(*noisy.shape) < prob / 2] = 0
        noisy[np.random.rand(*noisy.shape) < prob / 2] = 1
        return torch.tensor(noisy).float()

    def add_poisson_noise(self, image, scale=1.0):
        noisy = torch.poisson(image * scale) / scale
        return torch.clamp(noisy, 0, 1)

    def add_speckle_noise(self, image, mean=0, std=0.25):
        noise = torch.randn_like(image) * std + mean
        return torch.clamp(image * (1 + noise), 0, 1)

    def apply_noise(self, image):
        if self.config.noise_type == 'gaussian':
            return self.add_gaussian_noise(image, std=self.config.noise_std)
        elif self.config.noise_type == 'salt_pepper':
            return self.add_salt_and_pepper_noise(image, prob=self.config.salt_pepper_prob)
        elif self.config.noise_type == 'poisson':
            return self.add_poisson_noise(image, scale=self.poisson_scale)
        elif self.config.noise_type == 'speckle':
            return self.add_speckle_noise(image, std=self.config.speckle_std)
        return image

    # ---- TransformProcess 与原始一致 ----
    def TransformProcess(self, images, alpha):
        transformed_images = torch.zeros(
            size=(images.shape[0], 15, images.shape[2], images.shape[3]),
            dtype=torch.float32, device=config.device
        )
        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=config.device)
        alpha_63 = alpha_tensor * 63.0
        half_alpha_63 = alpha_63 / 2.0

        boundaries = [
            half_alpha_63, 63.0 - half_alpha_63,
            63.0 + half_alpha_63, 127.0 - half_alpha_63,
            127.0 + half_alpha_63, 191.0 - half_alpha_63,
            191.0 + half_alpha_63, 255.0 - half_alpha_63
        ]
        zero = torch.tensor(0.0, dtype=torch.float32, device=config.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=config.device)
        max_val = torch.tensor(255.0, dtype=torch.float32, device=config.device)
        c64 = torch.tensor(64.0, dtype=torch.float32, device=config.device)
        c63 = torch.tensor(63.0, dtype=torch.float32, device=config.device)
        c127 = torch.tensor(127.0, dtype=torch.float32, device=config.device)
        c191 = torch.tensor(191.0, dtype=torch.float32, device=config.device)

        for channel in range(3):
            pixel = images[:, channel, :, :].to(torch.float32)
            s1 = one - (pixel - half_alpha_63) / (c64 - alpha_63)
            s2 = one - s1
            s3 = one - (pixel - c63 - half_alpha_63) / (c64 - alpha_63)
            s4 = one - s3
            s5 = one - (pixel - c127 - half_alpha_63) / (c64 - alpha_63)
            s6 = one - s5
            s7 = one - (pixel - c191 - half_alpha_63) / (c64 - alpha_63)
            s8 = one - s7

            transformed_images[:, 0 + channel * 5] = torch.where(
                (pixel >= zero) & (pixel <= boundaries[0]), one,
                transformed_images[:, 0 + channel * 5])
            transformed_images[:, 0 + channel * 5] = torch.where(
                (pixel >= boundaries[0]) & (pixel <= boundaries[1]), s1,
                transformed_images[:, 0 + channel * 5])
            transformed_images[:, 1 + channel * 5] = torch.where(
                (pixel >= boundaries[0]) & (pixel <= boundaries[1]), s2,
                transformed_images[:, 1 + channel * 5])
            transformed_images[:, 1 + channel * 5] = torch.where(
                (pixel >= boundaries[1]) & (pixel <= boundaries[2]), one,
                transformed_images[:, 1 + channel * 5])
            transformed_images[:, 1 + channel * 5] = torch.where(
                (pixel >= boundaries[2]) & (pixel <= boundaries[3]), s3,
                transformed_images[:, 1 + channel * 5])
            transformed_images[:, 2 + channel * 5] = torch.where(
                (pixel >= boundaries[2]) & (pixel <= boundaries[3]), s4,
                transformed_images[:, 2 + channel * 5])
            transformed_images[:, 2 + channel * 5] = torch.where(
                (pixel >= boundaries[3]) & (pixel <= boundaries[4]), one,
                transformed_images[:, 2 + channel * 5])
            transformed_images[:, 2 + channel * 5] = torch.where(
                (pixel >= boundaries[4]) & (pixel <= boundaries[5]), s5,
                transformed_images[:, 2 + channel * 5])
            transformed_images[:, 3 + channel * 5] = torch.where(
                (pixel >= boundaries[4]) & (pixel <= boundaries[5]), s6,
                transformed_images[:, 3 + channel * 5])
            transformed_images[:, 3 + channel * 5] = torch.where(
                (pixel >= boundaries[5]) & (pixel <= boundaries[6]), one,
                transformed_images[:, 3 + channel * 5])
            transformed_images[:, 3 + channel * 5] = torch.where(
                (pixel >= boundaries[6]) & (pixel <= boundaries[7]), s7,
                transformed_images[:, 3 + channel * 5])
            transformed_images[:, 4 + channel * 5] = torch.where(
                (pixel >= boundaries[6]) & (pixel <= boundaries[7]), s8,
                transformed_images[:, 4 + channel * 5])
            transformed_images[:, 4 + channel * 5] = torch.where(
                (pixel >= boundaries[7]) & (pixel <= max_val), one,
                transformed_images[:, 4 + channel * 5])
        return transformed_images

    def load_preprocess_image(self, image, alpha):
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((64, 64))
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_tensor = self.apply_noise(image_tensor)
        image = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        image = self.add_contrast_enhancement(image)
        image = self.add_sharpening(image)
        image = image.resize((32, 32))
        image = np.array(image)
        image = image.reshape((32, 32, 3))
        return image

    def add_contrast_enhancement(self, image):
        img_array = np.array(image)
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

    def add_sharpening(self, image):
        img_array = np.array(image)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(sharpened)

    # ---- 缓存文件名 ----
    def _noise_cache_id(self):
        nt = self.config.noise_type
        if nt == 'gaussian':
            return f"gaussian_std{self.config.noise_std:.5f}"
        elif nt == 'salt_pepper':
            return f"saltpepper_prob{self.config.salt_pepper_prob:.5f}"
        elif nt == 'poisson':
            return f"poisson_scale{self.poisson_scale:.5f}"
        elif nt == 'speckle':
            return f"speckle_std{self.config.speckle_std:.5f}"
        return nt

    # ============================================================
    # ★ 关键修复 1: 原子保存 + 加载容错 + 形状校验
    # ============================================================
    def process_and_save_images(self, images, file_name, alpha):
        if not os.path.exists(self.preprocessed_data_path):
            os.makedirs(self.preprocessed_data_path)

        name, ext = os.path.splitext(file_name)
        noise_id = self._noise_cache_id()
        seed_tag = f"_seed{self.seed}" if self.seed is not None else ""
        file_name = f"{name}_{noise_id}{seed_tag}{ext}"
        processed_images_file = os.path.join(self.preprocessed_data_path, file_name)

        # -------- (a) 尝试加载已有缓存，损坏则删除并重建 --------
        if os.path.exists(processed_images_file):
            try:
                data = torch.load(processed_images_file, map_location='cpu')
                # 形状与 dtype 校验
                if (isinstance(data, torch.Tensor)
                        and data.dim() == 4
                        and data.shape[0] == len(images)
                        and data.shape[1] == 3):
                    print(f"{file_name} already exists. Loading the data.")
                    return data.to(config.device)
                else:
                    print(f"{file_name} 缓存内容异常，删除后重新生成。")
                    os.remove(processed_images_file)
            except Exception as e:
                print(f"{file_name} 缓存损坏，删除后重新生成。原因: {e}")
                try:
                    os.remove(processed_images_file)
                except OSError:
                    pass

        # -------- (b) 重新生成，先写 .tmp 再原子重命名 --------
        print(f"Processing {file_name} and saving.")
        processed_images = np.array([self.load_preprocess_image(img, alpha) for img in images])
        processed_images = torch.from_numpy(processed_images).float().permute(0, 3, 1, 2)

        tmp_path = processed_images_file + ".tmp"
        torch.save(processed_images, tmp_path)
        os.replace(tmp_path, processed_images_file)   # POSIX 原子操作

        return processed_images.to(config.device)

    def load_cifar10_data(self, alpha):
        logger.info("Load the CIFAR-10 dataset...")
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

        train_images = train_dataset.data
        train_labels = train_dataset.targets
        test_images = test_dataset.data
        test_labels = test_dataset.targets

        processed_train_images = self.process_and_save_images(train_images, 'processed_train_images.pt', alpha)
        processed_test_images = self.process_and_save_images(test_images, 'processed_test_images.pt', alpha)

        transformed_train_images = self.TransformProcess(processed_train_images, alpha)
        transformed_test_images = self.TransformProcess(processed_test_images, alpha)

        train_labels = torch.tensor(train_labels, dtype=torch.long, device=config.device)
        test_labels = torch.tensor(test_labels, dtype=torch.long, device=config.device)

        assert not torch.isnan(transformed_train_images).any()
        assert not torch.isnan(transformed_test_images).any()
        assert not torch.isinf(transformed_train_images).any()
        assert not torch.isinf(transformed_test_images).any()

        logger.info("CIFAR-10 数据加载及预处理完成")
        return transformed_train_images, train_labels, transformed_test_images, test_labels

    def train_model(self, transformed_train_images, train_labels, transformed_test_images, test_labels):
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
        logger.info(f"开始训练. epochs={self.config.epochs}, batch_size={self.config.batch_size}")

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

            self.model.eval()
            all_targets = []
            all_predictions = []
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
                        f"TrainLoss: {avg_train_loss:.4f} - TrainAcc: {accuracy_train:.2f}% - "
                        f"ValLoss: {avg_val_loss:.4f} - ValAcc: {accuracy_val:.2f}% - "
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
            else:
                patience_count += 1

            if accuracy_val > best_val_accuracy:
                best_val_accuracy = accuracy_val
                best_precision = precision
                best_recall = recall
                best_f1_score = f1_score

            if patience_count >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        model_path = os.path.join(config.save_path, f"best_model_{best_epoch}.pth")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'val_accuracy': best_val_accuracy,
            'training_history': training_history
        }, model_path)

        total_time = time.time() - start_time
        logger.info(f'训练完成 - Total: {total_time:.2f}s - BestEpoch: {best_epoch} - '
                    f'BestValLoss: {best_val_loss:.4f} - BestValAcc: {best_val_accuracy:.2f}% - '
                    f'P: {best_precision:.4f} - R: {best_recall:.4f} - F1: {best_f1_score:.4f}')

        self.model.load_state_dict(best_model_state)
        return best_val_accuracy, best_model_state, training_history

    def evaluate_model(self, transformed_test_images, test_labels):
        self.model.eval()
        all_probs = []
        all_targets = []

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
        except Exception:
            auc_roc = 0.5

        logloss = log_loss(all_targets, all_probs)
        cm = confusion_matrix(all_targets, predictions)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        print(f'准确率: {accuracy:.4f}')
        print(f'AUC-ROC: {auc_roc:.4f}')
        print(f'对数损失: {logloss:.4f}')
        print(f'各类别准确率: {class_accuracies}')

        return accuracy, auc_roc, logloss, class_accuracies

# ============================================================
# 单次完整实验（可复现）
# ============================================================
def run_experiment(cfg: Config, seed: int):
    """给定配置 + 随机种子，运行一次完整实验并返回指标字典。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    processor = ImageProcessor(cfg, seed=seed)
    transformed_train_images, train_labels, transformed_test_images, test_labels = processor.load_cifar10_data(0.0)

    best_val_accuracy, _, _ = processor.train_model(
        transformed_train_images, train_labels, transformed_test_images, test_labels
    )

    accuracy, auc_roc, logloss, class_acc = processor.evaluate_model(
        transformed_test_images, test_labels
    )

    composite_score = accuracy * 0.4 + auc_roc * 0.3 + (1 - logloss) * 0.3

    return {
        'seed': seed,
        'accuracy': float(accuracy),
        'auc_roc': float(auc_roc),
        'logloss': float(logloss),
        'composite_score': float(composite_score),
        'best_val_accuracy': float(best_val_accuracy),
        'class_accuracies': class_acc.tolist(),
    }

def apply_noise_params_to_config(cfg: Config, params: dict):
    cfg.noise_type = params['noise_type']
    if cfg.noise_type == 'gaussian':
        cfg.noise_std = params.get('noise_std', cfg.noise_std)
    elif cfg.noise_type == 'salt_pepper':
        cfg.salt_pepper_prob = params.get('salt_pepper_prob', cfg.salt_pepper_prob)
    elif cfg.noise_type == 'poisson':
        cfg.poisson_scale = params.get('poisson_scale', cfg.poisson_scale)
    elif cfg.noise_type == 'speckle':
        cfg.speckle_std = params.get('speckle_std', cfg.speckle_std)

# ============================================================
# Optuna objective
# ============================================================
def objective(trial):
    try:
        cfg = Config()

        noise_type = trial.suggest_categorical(
            'noise_type', ['gaussian', 'salt_pepper', 'poisson', 'speckle'])

        params = {'noise_type': noise_type}
        if noise_type == 'gaussian':
            params['noise_std'] = trial.suggest_float('noise_std', 0.01, 0.3)
        elif noise_type == 'salt_pepper':
            params['salt_pepper_prob'] = trial.suggest_float('salt_pepper_prob', 0.01, 0.1)
        elif noise_type == 'poisson':
            params['poisson_scale'] = trial.suggest_float('poisson_scale', 0.1, 5.0)
        elif noise_type == 'speckle':
            params['speckle_std'] = trial.suggest_float('speckle_std', 0.01, 0.5)

        apply_noise_params_to_config(cfg, params)
        logger.info(f"Trial {trial.number}: {params}")

        result = run_experiment(cfg, seed=42)
        logger.info(f"Trial {trial.number} - 综合评分: {result['composite_score']:.4f} "
                    f"(acc={result['accuracy']:.4f}, auc={result['auc_roc']:.4f}, logloss={result['logloss']:.4f})")

        trial.report(result['composite_score'], step=0)
        return result['composite_score']

    except TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} 失败: {e}")
        return 0.0

# ============================================================
# ★ 关键修复 2: 重复实验加"失败重试 + 换种子 + 清缓存"
# ============================================================
def _clear_cache_for_seed(seed, cache_dir='./processed_data'):
    """删除与 seed 相关的缓存文件，防止下次继续撞损坏文件。"""
    if not os.path.isdir(cache_dir):
        return
    for fn in os.listdir(cache_dir):
        if f"_seed{seed}" in fn:
            try:
                os.remove(os.path.join(cache_dir, fn))
                logger.info(f"已清理损坏缓存: {fn}")
            except OSError:
                pass

def run_repeated_evaluation(best_params: dict, n_repeats: int = 5,
                            base_seed: int = 42, max_retry: int = 3):
    """用最佳参数跑 n_repeats 次（不同 seed），失败自动重试。"""
    logger.info("=" * 70)
    logger.info(f"开始 {n_repeats} 次重复实验 (best_params={best_params})")
    logger.info("=" * 70)

    all_results = []
    for i in range(n_repeats):
        base = base_seed + i
        ok = False
        for attempt in range(max_retry):
            # 重试时换种子，避免一直撞同一个坏缓存
            seed = base + attempt * 1000
            logger.info(f"--- 重复 {i + 1}/{n_repeats} 第 {attempt + 1}/{max_retry} 次尝试 (seed={seed}) ---")

            cfg = Config()
            apply_noise_params_to_config(cfg, best_params)

            try:
                res = run_experiment(cfg, seed=seed)
                all_results.append(res)
                logger.info(f"[Repeat {i+1}] seed={seed} acc={res['accuracy']:.4f} "
                            f"auc={res['auc_roc']:.4f} logloss={res['logloss']:.4f} "
                            f"composite={res['composite_score']:.4f}")
                ok = True
                break
            except Exception as e:
                logger.error(f"重复 {i + 1} (seed={seed}) 第 {attempt + 1} 次失败: {e}")
                # 清理本次种子对应的缓存，避免重试又踩同一个坏文件
                _clear_cache_for_seed(seed)

        if not ok:
            logger.error(f"重复 {i + 1} 重试 {max_retry} 次仍失败，跳过。")

    return all_results

# ============================================================
# ★ 关键修复 3: n < 2 时 std 显示 nan 并给出 warning
# ============================================================
def summarize_results(results, metrics=None):
    """计算并打印均值 ± 标准差。n < 2 时 std 为 nan。"""
    if metrics is None:
        metrics = ['accuracy', 'auc_roc', 'logloss', 'composite_score', 'best_val_accuracy']

    n = len(results)
    summary = {}
    logger.info("=" * 70)
    logger.info(f"最终结果汇总 (n={n} 次重复):")
    if n < 2:
        logger.warning("有效重复次数不足 2，标准差无法计算（显示为 nan），请检查是否有 run 失败。")

    for m in metrics:
        vals = np.array([r[m] for r in results], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else float('nan')
        ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else float('nan')
        summary[m] = {'mean': mean, 'std': std, 'ci95': ci95,
                      'n': n, 'values': vals.tolist()}
        logger.info(f"  {m:>20s}: {mean:.4f} ± {std:.4f}  (95% CI ±{ci95:.4f}, n={n})")
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print(f"最终结果汇总 (n={n} 次重复):")
    for m in metrics:
        vals = np.array([r[m] for r in results], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else float('nan')
        print(f"  {m:>20s}: {mean:.4f} ± {std:.4f}  (n={n})")
    print("=" * 70)

    return summary

# ============================================================
# 可视化（可选）
# ============================================================
def visualize_blur_effects(images, noise_types):
    fig, axes = plt.subplots(len(noise_types), 2, figsize=(10, 15))
    for i, noise_type in enumerate(noise_types):
        axes[i, 0].imshow(images[0])
        axes[i, 0].set_title(f'Original - {noise_type}')
        axes[i, 0].axis('off')

        image_tensor = torch.tensor(images[0], dtype=torch.float32).permute(2, 0, 1) / 255.0
        processor = ImageProcessor(Config())
        if noise_type == 'gaussian':
            noisy_img = processor.add_gaussian_noise(image_tensor, std=0.1)
        elif noise_type == 'salt_pepper':
            noisy_img = processor.add_salt_and_pepper_noise(image_tensor, prob=0.05)
        elif noise_type == 'poisson':
            noisy_img = processor.add_poisson_noise(image_tensor, scale=1.0)
        elif noise_type == 'speckle':
            noisy_img = processor.add_speckle_noise(image_tensor, std=0.25)

        noisy_img = (noisy_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        axes[i, 1].imshow(noisy_img)
        axes[i, 1].set_title(f'{noise_type} Noise')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('noise_effects_comparison.png')
    plt.close()

# ============================================================
# 主流程
# ============================================================
def main():
    N_TRIALS = 10
    N_REPEATS = 5
    BASE_SEED = 42

    # ---------- 1. Optuna 搜索 ----------
    logger.info("=" * 70)
    logger.info(f"开始 Optuna 搜索: n_trials={N_TRIALS}")
    logger.info("=" * 70)

    sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        study_name=f"ode_noise_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    def log_study(study, trial):
        logger.info(f"Trial {trial.number} finished - Value: {trial.value}")

    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_study])

    logger.info("=" * 70)
    logger.info("Optuna 搜索完成")
    logger.info(f"最佳 trial: {study.best_trial.number}")
    logger.info(f"最佳综合评分: {study.best_value:.4f}")
    logger.info(f"最佳参数: {study.best_params}")
    logger.info("=" * 70)

    optuna_result_path = "./models/optuna_best_params.json"
    os.makedirs(os.path.dirname(optuna_result_path), exist_ok=True)
    with open(optuna_result_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_value': float(study.best_value),
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Optuna 最佳结果已保存到 {optuna_result_path}")

    # ---------- 2. N 次重复（带重试） ----------
    repeat_results = run_repeated_evaluation(
        study.best_params, n_repeats=N_REPEATS, base_seed=BASE_SEED, max_retry=3
    )

    # ---------- 3. 汇总 均值 ± 标准差 ----------
    if len(repeat_results) == 0:
        logger.warning("重复实验没有成功的运行。")
        return

    summary = summarize_results(repeat_results)

    summary_path = "./models/final_repeat_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'best_params': study.best_params,
            'n_repeats_requested': N_REPEATS,
            'n_repeats_successful': len(repeat_results),
            'per_run': repeat_results,
            'summary_mean_std': summary,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"{N_REPEATS} 次重复实验汇总已保存到 {summary_path}")

if __name__ == "__main__":
    main()
    logger.info("The program execution is completed.")

    # 最终结果汇总 (n=5 次重复):
    #              accuracy: 0.5834 ± 0.0121  (95% CI ±0.0106, n=5)
    #               auc_roc: 0.9210 ± 0.0065  (95% CI ±0.0057, n=5)
    #               logloss: 1.1713 ± 0.0230  (95% CI ±0.0202, n=5)
    #       composite_score: 0.4586 ± 0.0098  (95% CI ±0.0086, n=5)
    #     best_val_accuracy: 58.34 ± 1.21      (95% CI ±1.06, n=5)