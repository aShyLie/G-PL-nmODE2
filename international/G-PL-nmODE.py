import numpy as np
import os
import torch
from torchdiffeq import odeint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.functional as F
from PIL import ImageFilter, Image
import torch.nn as nn
import cv2
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import time
import optuna
from optuna.pruners import MedianPruner
from optuna.exceptions import TrialPruned
import random
import logging
from datetime import datetime
import json
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import defaultdict

# ======================================================================
# 2026-09-25 13:32:47,069 - INFO - Aggregated results over 5 runs (base_seed=42):
# 2026-09-25 13:32:47,069 - INFO - ----------------------------------------------------------------------
# 2026-09-25 13:32:47,069 - INFO -   Run  1  (seed= 42): acc= 51.95%  sens=0.4394  spec=0.9101  f1=0.4145
# 2026-09-25 13:32:47,069 - INFO -   Run  2  (seed= 43): acc= 55.19%  sens=0.4168  spec=0.9153  f1=0.4154
# 2026-09-25 13:32:47,070 - INFO -   Run  3  (seed= 44): acc= 55.19%  sens=0.4456  spec=0.9134  f1=0.4102
# 2026-09-25 13:32:47,070 - INFO -   Run  4  (seed= 45): acc= 62.99%  sens=0.5217  spec=0.9298  f1=0.5087
# 2026-09-25 13:32:47,070 - INFO -   Run  5  (seed= 46): acc= 56.49%  sens=0.4235  spec=0.9157  f1=0.4319
# 2026-09-25 13:32:47,070 - INFO - ----------------------------------------------------------------------
# 2026-09-25 13:32:47,070 - INFO -   Accuracy    : 56.36% ± 4.07%   (95% CI ±3.56%)
# 2026-09-25 13:32:47,070 - INFO -   Sensitivity : 0.4494 ± 0.0421
# 2026-09-25 13:32:47,070 - INFO -   Specificity : 0.9168 ± 0.0076
# 2026-09-25 13:32:47,071 - INFO -   F1-score    : 0.4361 ± 0.0414
# 2026-09-25 13:32:47,071 - INFO - ======================================================================


# ============================================================
# 统一随机种子
# ============================================================
def set_seed(seed):
    """统一的种子设置，用于每次 run 重新播种"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 保留原有的初始播种（兼容旧行为）
set_seed(42)


# 配置日志系统
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


# 数据增强控制器类
class DataAugmentationController:
    """数据增强控制器，支持手动控制多个数据增强操作"""

    def __init__(self, augmentation_config=None):
        self.default_config = {
            'random_rotation': False,
            'random_horizontal_flip': False,
            'random_vertical_flip': False,
            'random_crop': False,
            'color_jitter': False,
            'gaussian_blur': False,
            'gaussian_noise': False,
            'elastic_transform': False,
            'random_affine': False,
            'random_brightness': False,
            'random_contrast': False,
            'random_saturation': False,
            'random_hue': False,
            'random_sharpness': False,
            'random_erasing': False,
            'random_perspective': False,
            'random_posterize': False,
            'random_solarize': False,
            'random_adjust_sharpness': False,
            'random_autocontrast': False,
            'random_equalize': False,
            'speckle_noise': False,
            'salt_pepper_noise': False,
            'motion_blur': False,
        }

        self.config = self.default_config.copy()
        if augmentation_config:
            self.config.update(augmentation_config)

        logger.info(f"数据增强控制器初始化完成，当前配置: {self.get_enabled_augmentations()}")

    def get_enabled_augmentations(self):
        return [name for name, enabled in self.config.items() if enabled]

    def enable_augmentation(self, augmentation_name):
        if augmentation_name in self.config:
            self.config[augmentation_name] = True
            logger.info(f"启用数据增强: {augmentation_name}")
        else:
            logger.warning(f"未知的数据增强: {augmentation_name}")

    def disable_augmentation(self, augmentation_name):
        if augmentation_name in self.config:
            self.config[augmentation_name] = False
            logger.info(f"禁用数据增强: {augmentation_name}")
        else:
            logger.warning(f"未知的数据增强: {augmentation_name}")

    def set_config(self, new_config):
        self.config.update(new_config)
        logger.info(f"更新数据增强配置: {self.get_enabled_augmentations()}")

    def apply_augmentations(self, image, is_training=True):
        if not is_training:
            return image

        if not any(self.config.values()):
            return image

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        original_shape = image.shape
        original_dtype = image.dtype

        if image.shape == (1, 1, 1) or image.shape == (1, 1):
            logger.warning(f"图像太小，跳过数据增强: {image.shape}")
            if len(original_shape) == 3 and original_shape[2] == 1:
                return np.zeros((256, 256, 1), dtype=np.uint8)
            elif len(original_shape) == 2:
                return np.zeros((256, 256), dtype=np.uint8)
            else:
                return image

        if image.shape[0] < 8 or image.shape[1] < 8:
            logger.warning(f"图像太小 ({image.shape})，跳过数据增强")
            if len(image.shape) == 2:
                image = cv2.resize(image, (8, 8))
            elif len(image.shape) == 3:
                image = cv2.resize(image, (8, 8))
            else:
                return image

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image_min = image.min()
                image_max = image.max()
                if image_max > 255 or image_min < 0:
                    image = np.clip(image, 0, 255)
                image = image.astype(np.uint8)

        try:
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image, mode='L')
            elif len(image.shape) == 3 and image.shape[2] == 1:
                pil_image = Image.fromarray(image[:, :, 0], mode='L')
            elif len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image, mode='RGB')
            else:
                logger.warning(f"不支持的图像形状: {image.shape}，尝试转换为2D")
                if len(image.shape) == 3:
                    image = image[:, :, 0]
                pil_image = Image.fromarray(image, mode='L')
        except Exception as e:
            logger.error(f"图像转换失败: {e}, 形状: {image.shape}, dtype: {image.dtype}")
            return image

        try:
            img_np = np.array(pil_image)
        except Exception as e:
            logger.error(f"PIL转numpy失败: {e}")
            return image

        if img_np.shape[0] < 8 or img_np.shape[1] < 8:
            logger.warning(f"图像太小 ({img_np.shape})，调整到8x8")
            if len(img_np.shape) == 2:
                img_np = cv2.resize(img_np, (8, 8))
            elif len(img_np.shape) == 3:
                img_np = cv2.resize(img_np, (8, 8))
            else:
                return img_np

        try:
            if self.config['random_rotation']:
                img_np = self.random_rotation(img_np)
            if self.config['random_horizontal_flip']:
                img_np = self.random_horizontal_flip(img_np)
            if self.config['random_vertical_flip']:
                img_np = self.random_vertical_flip(img_np)
            if self.config['random_crop']:
                img_np = self.random_crop(img_np)
            if self.config['gaussian_blur']:
                img_np = self.gaussian_blur(img_np)
            if self.config['gaussian_noise']:
                if img_np.shape[0] >= 8 and img_np.shape[1] >= 8:
                    img_np = self.add_gaussian_noise(img_np)
                else:
                    logger.warning(f"图像太小，跳过高斯噪声: {img_np.shape}")
            if self.config['elastic_transform']:
                img_np = self.elastic_transform(img_np)
            if self.config['random_affine']:
                img_np = self.random_affine_transform(img_np)
            if self.config['random_brightness']:
                img_np = self.random_brightness(img_np)
            if self.config['random_contrast']:
                img_np = self.random_contrast(img_np)
            if self.config['color_jitter'] and img_np.shape[-1] == 3:
                img_np = self.color_jitter(img_np)
            if self.config['speckle_noise']:
                img_np = self.add_speckle_noise(img_np)
            if self.config['salt_pepper_noise']:
                img_np = self.add_salt_pepper_noise(img_np)
            if self.config['motion_blur']:
                img_np = self.motion_blur(img_np)
        except Exception as e:
            logger.error(f"数据增强过程中出错: {e}")

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if len(original_shape) == 3 and original_shape[2] == 1 and len(img_np.shape) == 2:
            img_np = img_np[:, :, np.newaxis]

        if img_np.shape[0] < 8 or img_np.shape[1] < 8:
            logger.warning(f"增强后图像太小 ({img_np.shape})，调整到8x8")
            if len(img_np.shape) == 2:
                img_np = cv2.resize(img_np, (8, 8))
            elif len(img_np.shape) == 3:
                img_np = cv2.resize(img_np, (8, 8))

        return img_np

    def random_rotation(self, image, angle_range=(-15, 15)):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def random_horizontal_flip(self, image, p=0.5):
        if random.random() < p:
            return cv2.flip(image, 1)
        return image

    def random_vertical_flip(self, image, p=0.5):
        if random.random() < p:
            return cv2.flip(image, 0)
        return image

    def random_crop(self, image, crop_size=(224, 224)):
        h, w = image.shape[:2]
        if h < 8 or w < 8:
            return image
        crop_h = min(crop_size[0], h)
        crop_w = min(crop_size[1], w)
        if crop_h < 8 or crop_w < 8:
            return image
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped = image[top:top + crop_h, left:left + crop_w]
        if cropped.shape[:2] != (h, w):
            cropped = cv2.resize(cropped, (w, h))
        return cropped

    def gaussian_blur(self, image, kernel_size=(3, 3), sigma_range=(0.1, 2.0)):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        return cv2.GaussianBlur(image, kernel_size, sigma)

    def add_gaussian_noise(self, image, mean=0, std_range=(0.01, 0.05)):
        if image.shape == (1, 1, 1) or image.shape == (1, 1):
            return image
        if len(image.shape) == 2:
            if image.shape[0] < 8 or image.shape[1] < 8:
                return image
        elif len(image.shape) == 3:
            if image.shape[0] < 8 or image.shape[1] < 8:
                return image
        std = random.uniform(std_range[0], std_range[1])
        try:
            gauss = np.random.normal(mean, std, image.shape) * 255
            return image + gauss
        except Exception as e:
            logger.error(f"生成高斯噪声失败: {e}")
            return image

    def elastic_transform(self, image, alpha_range=(30, 36), sigma_range=(4, 6)):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        sigma = random.uniform(sigma_range[0], sigma_range[1])
        random_state = np.random.RandomState(None)
        shape = image.shape[:2]
        dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        if len(image.shape) == 3:
            transformed = []
            for i in range(image.shape[2]):
                channel = cv2.remap(image[:, :, i], indices[1].astype(np.float32),
                                    indices[0].astype(np.float32), cv2.INTER_LINEAR)
                transformed.append(channel)
            transformed = np.stack(transformed, axis=2)
        else:
            transformed = cv2.remap(image, indices[1].astype(np.float32),
                                    indices[0].astype(np.float32), cv2.INTER_LINEAR)
        return transformed

    def random_affine_transform(self, image, scale_range=(0.9, 1.1),
                                translate_range=(-0.1, 0.1), shear_range=(-5, 5)):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        h, w = image.shape[:2]
        scale = random.uniform(scale_range[0], scale_range[1])
        tx = random.uniform(translate_range[0], translate_range[1]) * w
        ty = random.uniform(translate_range[0], translate_range[1]) * h
        shear = random.uniform(shear_range[0], shear_range[1])
        M = np.float32([[scale, shear, tx], [0, scale, ty]])
        return cv2.warpAffine(image, M, (w, h))

    def random_brightness(self, image, brightness_range=(-30, 30)):
        brightness = random.randint(brightness_range[0], brightness_range[1])
        return cv2.add(image, brightness)

    def random_contrast(self, image, contrast_range=(0.8, 1.2)):
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        return cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    def color_jitter(self, image, brightness_range=(-30, 30), contrast_range=(0.8, 1.2),
                     saturation_range=(0.8, 1.2), hue_range=(-10, 10)):
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        if random.random() < 0.8:
            brightness = random.randint(brightness_range[0], brightness_range[1])
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness, 0, 255)
        if random.random() < 0.8:
            saturation = random.uniform(saturation_range[0], saturation_range[1])
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        if random.random() < 0.8:
            hue = random.randint(hue_range[0], hue_range[1])
            hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def add_speckle_noise(self, image, noise_level_range=(0.01, 0.05)):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
        noise = np.random.randn(*image.shape) * noise_level * 255
        return image + image * noise

    def add_salt_pepper_noise(self, image, salt_prob=0.01, pepper_prob=0.01):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        noisy = np.copy(image)
        salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
        if len(image.shape) == 3:
            noisy[salt_mask] = [255, 255, 255]
        else:
            noisy[salt_mask] = 255
        pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob
        if len(image.shape) == 3:
            noisy[pepper_mask] = [0, 0, 0]
        else:
            noisy[pepper_mask] = 0
        return noisy

    def motion_blur(self, image, kernel_size_range=(3, 7)):
        if image.shape[0] < 8 or image.shape[1] < 8:
            return image
        kernel_size = random.randint(kernel_size_range[0], kernel_size_range[1])
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)


class JunQuZongData:
    def __init__(self, json_file_path, image_folder_path, augmentation_controller=None):
        self.json_file_path = json_file_path
        self.image_folder_path = image_folder_path
        self.image_category_map = {}
        self.categories = {}
        self.augmentation_controller = augmentation_controller

    def load_data(self):
        with open(self.json_file_path, 'r') as f:
            coco_data = json.load(f)
        self.categories = {category['id']: category['name'] for category in coco_data['categories']}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            image_filename = coco_data['images'][image_id]['file_name'].split('/')[-1]
            self.image_category_map[image_filename] = category_id

    def preprocess_image(self, image_path, apply_augmentation=True):
        try:
            img = load_img(image_path, target_size=(256, 256))
            img = img.convert('L')
            img_array = img_to_array(img)

            if img_array.shape != (256, 256, 1):
                if len(img_array.shape) == 2:
                    img_array = img_array[:, :, np.newaxis]
                elif img_array.shape[2] != 1:
                    img_array = img_array[:, :, 0:1]
                else:
                    if img_array.shape[0] != 256 or img_array.shape[1] != 256:
                        img_array = cv2.resize(img_array[:, :, 0], (256, 256))[:, :, np.newaxis]

            if apply_augmentation and self.augmentation_controller:
                try:
                    img_array = self.augmentation_controller.apply_augmentations(
                        img_array, is_training=True
                    )
                except Exception as e:
                    logger.error(f"数据增强失败: {image_path}, 错误: {e}")

            if img_array.shape[0] < 8 or img_array.shape[1] < 8:
                if len(img_array.shape) == 2:
                    img_array = cv2.resize(img_array, (256, 256))
                elif len(img_array.shape) == 3:
                    img_array = cv2.resize(img_array, (256, 256))

            if img_array.shape != (256, 256, 1):
                if len(img_array.shape) == 2:
                    img_array = img_array[:, :, np.newaxis]
                if img_array.shape[0] != 256 or img_array.shape[1] != 256:
                    img_array = cv2.resize(img_array[:, :, 0], (256, 256))[:, :, np.newaxis]

            return img_array

        except Exception as e:
            logger.error(f"图像预处理失败: {image_path}, 错误: {e}")
            return np.zeros((256, 256, 1), dtype=np.float32)

    def create_dataset(self, X_data, y_data, apply_augmentation=True):
        dataset = []
        for i in range(len(X_data)):
            filename = X_data[i]
            category_id = y_data[i]
            image_filepath = os.path.join(self.image_folder_path, filename)
            if os.path.exists(image_filepath) and filename in self.image_category_map:
                X = self.preprocess_image(image_filepath, apply_augmentation)
                y = category_id
                dataset.append({'image': X, 'label': y})
        return dataset


class BreastCancerDataset(Dataset):
    def __init__(self, data, augmentation_controller=None, is_training=True):
        self.data = data
        self.augmentation_controller = augmentation_controller
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        image = sample['image']
        label = sample['label']

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if image.shape == (1, 1, 1):
            image = np.zeros((256, 256, 1), dtype=np.float32)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        elif len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if image.shape[0] != 256 or image.shape[1] != 256:
            if len(image.shape) == 3:
                channels = image.shape[2]
                resized_channels = []
                for c in range(channels):
                    channel = cv2.resize(image[:, :, c], (256, 256))
                    resized_channels.append(channel)
                image = np.stack(resized_channels, axis=2)
            else:
                image = cv2.resize(image, (256, 256))
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]

        if len(image.shape) == 3 and image.shape[2] != 1:
            image = image[:, :, 0:1]

        if self.augmentation_controller and self.is_training and len(image.shape) in [2, 3]:
            try:
                image = self.augmentation_controller.apply_augmentations(image, is_training=True)
            except Exception as e:
                logger.error(f"数据增强失败 (索引 {idx}): {e}")

        if image.shape != (256, 256, 1):
            if len(image.shape) == 2:
                image = cv2.resize(image, (256, 256))[:, :, np.newaxis]
            elif len(image.shape) == 3:
                if image.shape[2] >= 1:
                    image = cv2.resize(image[:, :, 0], (256, 256))[:, :, np.newaxis]
                else:
                    image = np.zeros((256, 256, 1), dtype=np.float32)

        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return image_tensor, label_tensor


class Config:
    def __init__(self):
        self.epochs = 30
        self.batch_size = 64
        self.num_classes = 7
        self.input_shape = (256, 256, 1)
        self.layers_num = 5
        self.verbose = 2
        self.weight_decay = 1e-4

        self.num_workers = 0

        # 新增：随机种子（供 train_test_split 使用）
        self.seed = 42

        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        logger.info(f"device: {self.device}")

        self.logger_name = "train"

        self.best_epoch = 0
        self.best_acc = 0
        self.save_path = "./models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.xsize = 256 * 256 * 5
        self.ysize = 2048
        self.asize = 7
        self.alpha = 0.00001
        self.tbar = 2.0
        self.t0 = 0
        self.xi1 = 0.49
        self.xi2 = 0.49
        self.in_chans = 1

        self.json_file_path = r"C:\Users\Administrator\Desktop\暂存数据集的\US_data_Revised\crop_private_breast_US\internationalSet\internationalSet.json"
        self.image_folder_path = r"C:\Users\Administrator\Desktop\暂存数据集的\US_data_Revised\crop_private_breast_US\internationalSet\data"

        self.over_sample = {
            0: 19, 1: 11, 2: 8, 3: 19, 4: 23, 5: 28, 6: 99
        }

        self.augmentation_config = {
            'random_rotation': True,
            'random_horizontal_flip': True,
            'random_vertical_flip': False,
            'random_crop': False,
            'color_jitter': False,
            'gaussian_blur': True,
            'gaussian_noise': True,
            'elastic_transform': False,
            'random_affine': False,
            'random_brightness': True,
            'random_contrast': True,
            'random_saturation': False,
            'random_hue': False,
            'random_sharpness': False,
            'random_erasing': False,
            'random_perspective': False,
            'random_posterize': False,
            'random_solarize': False,
            'random_adjust_sharpness': False,
            'random_autocontrast': False,
            'random_equalize': False,
            'speckle_noise': False,
            'salt_pepper_noise': False,
            'motion_blur': False,
        }


config = Config()


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

    def save(self, path):
        torch.save({"w1": self.w1.weight, "b": self.w1.bias, "w2": self.w2.weight}, path)

    def load(self, path):
        params = torch.load(path, map_location=config.device)
        self.w1.weight = params["w1"]
        self.w2.weight = params["w2"]
        self.w1.bias = params["b"]


class ImageProcessor:
    def __init__(self, config, augmentation_controller=None):
        self.layer_num = config.layers_num
        self.config = config
        self.model = nmODE2()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=config.alpha, weight_decay=config.weight_decay)

        self.preprocessed_data_path = './processed_data'
        os.makedirs(self.preprocessed_data_path, exist_ok=True)

        self.augmentation_controller = augmentation_controller

        self.model.to(config.device)
        logger.info(f"The model initialization is complete and has been moved to the device: {config.device}")

    def TransformProcess(self, images, alpha):
        if images.dim() == 5:
            images = images.squeeze(-1)

        batch_size, channels, height, width = images.shape

        transformed_images = torch.zeros(size=(batch_size, 5, height, width),
                                         dtype=torch.float32, device=config.device)

        alpha_tensor = torch.tensor(alpha, dtype=torch.float32, device=config.device)
        alpha_63 = alpha_tensor * 63.0
        half_alpha_63 = alpha_63 / 2.0

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

        zero = torch.tensor(0.0, dtype=torch.float32, device=config.device)
        one = torch.tensor(1.0, dtype=torch.float32, device=config.device)
        max_val = torch.tensor(255.0, dtype=torch.float32, device=config.device)
        c64 = torch.tensor(64.0, dtype=torch.float32, device=config.device)
        c63 = torch.tensor(63.0, dtype=torch.float32, device=config.device)
        c127 = torch.tensor(127.0, dtype=torch.float32, device=config.device)
        c191 = torch.tensor(191.0, dtype=torch.float32, device=config.device)

        channel = 0
        pixel = images[:, channel, :, :].to(torch.float32)

        s1 = one - (pixel - half_alpha_63) / (c64 - alpha_63)
        s2 = one - s1
        s3 = one - (pixel - c63 - half_alpha_63) / (c64 - alpha_63)
        s4 = one - s3
        s5 = one - (pixel - c127 - half_alpha_63) / (c64 - alpha_63)
        s6 = one - s5
        s7 = one - (pixel - c191 - half_alpha_63) / (c64 - alpha_63)
        s8 = one - s7

        transformed_images[:, 0, :, :] = torch.where(
            (pixel >= zero) & (pixel <= boundaries[0]), one,
            transformed_images[:, 0, :, :])
        transformed_images[:, 0, :, :] = torch.where(
            (pixel >= boundaries[0]) & (pixel <= boundaries[1]), s1,
            transformed_images[:, 0, :, :])

        transformed_images[:, 1, :, :] = torch.where(
            (pixel >= boundaries[0]) & (pixel <= boundaries[1]), s2,
            transformed_images[:, 1, :, :])
        transformed_images[:, 1, :, :] = torch.where(
            (pixel >= boundaries[1]) & (pixel <= boundaries[2]), one,
            transformed_images[:, 1, :, :])
        transformed_images[:, 1, :, :] = torch.where(
            (pixel >= boundaries[2]) & (pixel <= boundaries[3]), s3,
            transformed_images[:, 1, :, :])

        transformed_images[:, 2, :, :] = torch.where(
            (pixel >= boundaries[2]) & (pixel <= boundaries[3]), s4,
            transformed_images[:, 2, :, :])
        transformed_images[:, 2, :, :] = torch.where(
            (pixel >= boundaries[3]) & (pixel <= boundaries[4]), one,
            transformed_images[:, 2, :, :])
        transformed_images[:, 2, :, :] = torch.where(
            (pixel >= boundaries[4]) & (pixel <= boundaries[5]), s5,
            transformed_images[:, 2, :, :])

        transformed_images[:, 3, :, :] = torch.where(
            (pixel >= boundaries[4]) & (pixel <= boundaries[5]), s6,
            transformed_images[:, 3, :, :])
        transformed_images[:, 3, :, :] = torch.where(
            (pixel >= boundaries[5]) & (pixel <= boundaries[6]), one,
            transformed_images[:, 3, :, :])
        transformed_images[:, 3, :, :] = torch.where(
            (pixel >= boundaries[6]) & (pixel <= boundaries[7]), s7,
            transformed_images[:, 3, :, :])

        transformed_images[:, 4, :, :] = torch.where(
            (pixel >= boundaries[6]) & (pixel <= boundaries[7]), s8,
            transformed_images[:, 4, :, :])
        transformed_images[:, 4, :, :] = torch.where(
            (pixel >= boundaries[7]) & (pixel <= max_val), one,
            transformed_images[:, 4, :, :])

        return transformed_images

    def load_breast_cancer_data(self):
        logger.info("Loading breast cancer dataset...")

        data_loader = JunQuZongData(config.json_file_path, config.image_folder_path,
                                    self.augmentation_controller)
        data_loader.load_data()

        X_data = list(data_loader.image_category_map.keys())
        y_data = list(data_loader.image_category_map.values())

        dataset = data_loader.create_dataset(X_data, y_data, apply_augmentation=False)

        # 关键：使用 self.config.seed，让数据划分随种子变化
        train_data, test_data = train_test_split(
            dataset, test_size=0.2, random_state=self.config.seed
        )

        for idx in range(len(train_data)):
            label = train_data[idx]['label']
            over_sample_count = config.over_sample.get(label, 1)
            for i in range(over_sample_count):
                train_data.append(train_data[idx])

        train_dataset = BreastCancerDataset(train_data, self.augmentation_controller, is_training=True)
        test_dataset = BreastCancerDataset(test_data, None, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers)

        logger.info("Breast cancer dataset loading and preprocessing completed")
        logger.info(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
        if self.augmentation_controller:
            logger.info(f"启用的数据增强: {self.augmentation_controller.get_enabled_augmentations()}")

        return train_loader, test_loader

    def calculate_metrics(self, y_true, y_pred, num_classes):
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)

        accuracy = np.sum(TP) / np.sum(cm)

        sensitivity = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
        specificity = np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0)
        f1_score = np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP, dtype=float),
                             where=(2 * TP + FP + FN) != 0)

        sensitivity_macro = np.mean(sensitivity)
        specificity_macro = np.mean(specificity)
        f1_score_macro = np.mean(f1_score)

        sensitivity_micro = np.sum(TP) / np.sum(TP + FN)
        specificity_micro = np.sum(TN) / np.sum(TN + FP)
        f1_score_micro = 2 * np.sum(TP) / (2 * np.sum(TP) + np.sum(FP) + np.sum(FN))

        return (accuracy, sensitivity_macro, specificity_macro, f1_score_macro,
                sensitivity_micro, specificity_micro, f1_score_micro)

    def train_model(self):
        train_loader, test_loader = self.load_breast_cancer_data()

        best_val_accuracy = 0.0
        best_epoch = 0
        best_val_loss = float('inf')
        best_model_state = None

        best_sensitivity_macro = 0.0
        best_specificity_macro = 0.0
        best_f1_score_macro = 0.0

        patience = 5
        min_delta = 0.001
        patience_count = 0

        training_history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        start_time = time.time()
        logger.info(f"Start training. Total number of epochs: {self.config.epochs}, "
                    f"Batch size: {self.config.batch_size}")

        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(config.device), target.to(config.device)
                data = self.TransformProcess(data, 0.5)

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

                if batch_idx % 100 == 0:
                    batch_acc = 100.0 * pred.eq(target.view_as(pred)).sum().item() / target.size(0)
                    logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - "
                                f"Batch {batch_idx}/{len(train_loader)} - "
                                f"Loss: {loss.item():.4f} - Acc: {batch_acc:.2f}%")

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
                    data = self.TransformProcess(data, 0.5)

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

            (accuracy, sensitivity_macro, specificity_macro, f1_score_macro,
             sensitivity_micro, specificity_micro, f1_score_micro) = self.calculate_metrics(
                all_targets, all_predictions, config.num_classes)

            training_history['epoch'].append(epoch + 1)
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_acc'].append(accuracy_train)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_acc'].append(accuracy_val)

            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - "
                        f"Training loss: {avg_train_loss:.4f} - "
                        f"Training accuracy: {accuracy_train:.2f}% - "
                        f"Verification loss: {avg_val_loss:.4f} - "
                        f"Verification accuracy: {accuracy_val:.2f}% - "
                        f"Sensitivity (Macro): {sensitivity_macro:.4f} - "
                        f"Specificity (Macro): {specificity_macro:.4f} - "
                        f"F1-score (Macro): {f1_score_macro:.4f} - "
                        f"Time: {time.time() - epoch_start_time:.2f}s")

            if avg_val_loss + min_delta < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_accuracy = accuracy_val
                best_sensitivity_macro = sensitivity_macro
                best_specificity_macro = specificity_macro
                best_f1_score_macro = f1_score_macro
                best_epoch = epoch + 1
                best_model_state = self.model.state_dict()
                patience_count = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}, "
                            f"accuracy: {best_val_accuracy:.2f}%")
            else:
                patience_count += 1

            if accuracy_val > best_val_accuracy:
                best_val_accuracy = accuracy_val
                best_sensitivity_macro = sensitivity_macro
                best_specificity_macro = specificity_macro
                best_f1_score_macro = f1_score_macro
                logger.info(f"New best validation accuracy: {best_val_accuracy:.2f}%")

            if patience_count >= patience:
                logger.info(f"Early stopping triggered epoch {epoch + 1}")
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
                    f'Best validation loss: {best_val_loss:.4f} - '
                    f'Best verification accuracy rate: {best_val_accuracy:.2f}% - '
                    f'Best sensitivity (Macro): {best_sensitivity_macro:.4f} - '
                    f'Best specificity (Macro): {best_specificity_macro:.4f} - '
                    f'Best F1-score (Macro): {best_f1_score_macro:.4f}')

        self.model.load_state_dict(best_model_state)

        # 返回完整指标字典
        return {
            'val_accuracy': best_val_accuracy,
            'sensitivity_macro': best_sensitivity_macro,
            'specificity_macro': best_specificity_macro,
            'f1_score_macro': best_f1_score_macro,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'model_state': best_model_state,
            'training_history': training_history,
        }

    def evaluate_model(self, test_loader):
        self.model.eval()
        total_test_loss = 0.0
        correct_test = 0
        total_test = 0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config.device), target.to(config.device)
                data = self.TransformProcess(data, 0.5)

                output = self.model(data)
                loss = self.criterion(output, target)
                total_test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct_test += pred.eq(target.view_as(pred)).sum().item()
                total_test += target.size(0)

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(pred.cpu().numpy().flatten())

        avg_test_loss = total_test_loss / total_test
        test_acc = 100.0 * correct_test / total_test

        (accuracy, sensitivity_macro, specificity_macro, f1_score_macro,
         sensitivity_micro, specificity_micro, f1_score_micro) = self.calculate_metrics(
            all_targets, all_predictions, config.num_classes)

        print(f'\nTest Loss: {avg_test_loss:.4f} - Test Accuracy: {test_acc:.2f}%')
        print(f'Test Sensitivity (Macro): {sensitivity_macro:.4f}, '
              f'Specificity (Macro): {specificity_macro:.4f}, '
              f'F1 Score (Macro): {f1_score_macro:.4f}')
        return test_acc, sensitivity_macro, specificity_macro, f1_score_macro


def objective(trial):
    try:
        config = Config()

        use_random_rotation = trial.suggest_categorical('use_random_rotation', [True, False])
        use_random_horizontal_flip = trial.suggest_categorical('use_random_horizontal_flip', [True, False])
        use_gaussian_blur = trial.suggest_categorical('use_gaussian_blur', [True, False])
        use_gaussian_noise = trial.suggest_categorical('use_gaussian_noise', [True, False])

        augmentation_config = {
            'random_rotation': use_random_rotation,
            'random_horizontal_flip': use_random_horizontal_flip,
            'gaussian_blur': use_gaussian_blur,
            'gaussian_noise': use_gaussian_noise,
        }

        augmentation_controller = DataAugmentationController(augmentation_config)
        processor = ImageProcessor(config, augmentation_controller)

        alpha = trial.suggest_float('alpha', 0.0, 1.0)
        logger.info(f"Trial {trial.number}: alpha = {alpha}")

        results = processor.train_model()
        best_val_accuracy = results['val_accuracy']

        logger.info(f"Trial {trial.number} finishes - "
                    f"Optimal verification accuracy: {best_val_accuracy:.2f}%")

        return best_val_accuracy
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        logger.exception("Detailed error message:")
        raise


def train_with_custom_augmentations(augmentation_config=None):
    config = Config()
    if augmentation_config is None:
        augmentation_config = config.augmentation_config

    augmentation_controller = DataAugmentationController(augmentation_config)
    processor = ImageProcessor(config, augmentation_controller)

    results = processor.train_model()
    return results['val_accuracy'], results['model_state'], results['training_history']


def example_custom_augmentations():
    logger.info("示例：手动控制数据增强组合")

    config1 = {
        'random_rotation': True, 'random_horizontal_flip': True,
        'random_vertical_flip': False, 'gaussian_blur': False,
        'gaussian_noise': False, 'random_brightness': False, 'random_contrast': False,
    }
    logger.info("训练配置1：只使用随机旋转和水平翻转")
    acc1, _, _ = train_with_custom_augmentations(config1)
    logger.info(f"配置1的验证准确率: {acc1:.2f}%")

    config2 = {
        'random_rotation': True, 'random_horizontal_flip': True,
        'random_vertical_flip': True, 'gaussian_blur': True,
        'gaussian_noise': True, 'random_brightness': True, 'random_contrast': True,
        'elastic_transform': False, 'random_affine': False,
    }
    logger.info("训练配置2：使用所有基本增强")
    acc2, _, _ = train_with_custom_augmentations(config2)
    logger.info(f"配置2的验证准确率: {acc2:.2f}%")

    config3 = {
        'random_rotation': False, 'random_horizontal_flip': False,
        'gaussian_blur': True, 'gaussian_noise': True,
        'speckle_noise': True, 'salt_pepper_noise': True,
    }
    logger.info("训练配置3：只使用噪声增强")
    acc3, _, _ = train_with_custom_augmentations(config3)
    logger.info(f"配置3的验证准确率: {acc3:.2f}%")

    logger.info("数据增强组合比较结果:")
    logger.info(f"配置1（旋转+翻转）: {acc1:.2f}%")
    logger.info(f"配置2（所有基本增强）: {acc2:.2f}%")
    logger.info(f"配置3（噪声增强）: {acc3:.2f}%")


# ============================================================
# 单次训练 / 多次运行聚合（新增）
# ============================================================
def run_single_training(augmentation_config, seed):
    """单次完整训练，返回指标字典"""
    global config

    set_seed(seed)

    # 每次运行都新建 Config，并覆盖全局 config（模型内部依赖全局 config）
    config = Config()
    config.seed = seed
    if augmentation_config is not None:
        config.augmentation_config = augmentation_config

    aug_controller = DataAugmentationController(augmentation_config)
    processor = ImageProcessor(config, aug_controller)

    results = processor.train_model()

    # 释放显存
    try:
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def run_multiple_experiments(runs, augmentation_config, base_seed=42):
    """重复 runs 次训练，输出均值 ± 标准差"""
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    histories = []

    for i in range(runs):
        seed = base_seed + i
        logger.info("\n" + "=" * 70)
        logger.info(f"RUN {i + 1}/{runs}  (seed={seed})")
        logger.info("=" * 70 + "\n")

        results = run_single_training(augmentation_config, seed)

        accuracies.append(results['val_accuracy'])
        sensitivities.append(results['sensitivity_macro'])
        specificities.append(results['specificity_macro'])
        f1_scores.append(results['f1_score_macro'])
        histories.append(results['training_history'])

        logger.info(f"[Run {i + 1}/{runs}] seed={seed} -> "
                    f"val_acc = {results['val_accuracy']:.2f}%, "
                    f"sens = {results['sensitivity_macro']:.4f}, "
                    f"spec = {results['specificity_macro']:.4f}, "
                    f"f1 = {results['f1_score_macro']:.4f}")

    def _stats(arr):
        arr = np.asarray(arr, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        ci95 = float(1.96 * std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, std, ci95

    acc_mean, acc_std, acc_ci = _stats(accuracies)
    sen_mean, sen_std, _ = _stats(sensitivities)
    spe_mean, spe_std, _ = _stats(specificities)
    f1_mean, f1_std, _ = _stats(f1_scores)

    logger.info("\n" + "=" * 70)
    logger.info(f"Aggregated results over {runs} runs (base_seed={base_seed}):")
    logger.info("-" * 70)
    for i in range(runs):
        logger.info(f"  Run {i + 1:2d}  (seed={base_seed + i:>3d}): "
                    f"acc={accuracies[i]:6.2f}%  "
                    f"sens={sensitivities[i]:.4f}  "
                    f"spec={specificities[i]:.4f}  "
                    f"f1={f1_scores[i]:.4f}")
    logger.info("-" * 70)
    logger.info(f"  Accuracy    : {acc_mean:.2f}% ± {acc_std:.2f}%   (95% CI ±{acc_ci:.2f}%)")
    logger.info(f"  Sensitivity : {sen_mean:.4f} ± {sen_std:.4f}")
    logger.info(f"  Specificity : {spe_mean:.4f} ± {spe_std:.4f}")
    logger.info(f"  F1-score    : {f1_mean:.4f} ± {f1_std:.4f}")
    logger.info("=" * 70 + "\n")

    # 保存汇总结果
    summary_path = os.path.join(
        config.save_path,
        f"runs_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'runs': runs,
                'base_seed': base_seed,
                'augmentation_config': augmentation_config,
                'accuracies': [float(x) for x in accuracies],
                'sensitivities': [float(x) for x in sensitivities],
                'specificities': [float(x) for x in specificities],
                'f1_scores': [float(x) for x in f1_scores],
                'accuracy_mean': acc_mean,
                'accuracy_std': acc_std,
                'accuracy_ci95': acc_ci,
                'sensitivity_mean': sen_mean,
                'sensitivity_std': sen_std,
                'specificity_mean': spe_mean,
                'specificity_std': spe_std,
                'f1_mean': f1_mean,
                'f1_std': f1_std,
            }, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")
    except Exception as e:
        logger.error(f"保存汇总失败: {e}")

    return {
        'accuracy_mean': acc_mean, 'accuracy_std': acc_std,
        'sensitivity_mean': sen_mean, 'sensitivity_std': sen_std,
        'specificity_mean': spe_mean, 'specificity_std': spe_std,
        'f1_mean': f1_mean, 'f1_std': f1_std,
        'accuracies': accuracies,
        'histories': histories,
    }


# ============================================================
# 主入口：支持 --runs / --seed / --aug / --no-aug / --use-optuna
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train neuronODE on breast US dataset with optional multi-run repeats"
    )
    parser.add_argument('--runs', type=int, default=1,
                        help='重复训练次数（每次用不同种子），默认 1')
    parser.add_argument('--seed', type=int, default=42,
                        help='基础随机种子，默认 42')
    parser.add_argument('--aug', action='store_true',
                        help='启用数据增强（非交互模式）')
    parser.add_argument('--no-aug', action='store_true',
                        help='关闭数据增强（非交互模式）')
    parser.add_argument('--use-optuna', action='store_true',
                        help='启用 Optuna 搜索（与 --runs > 1 互斥）')
    args = parser.parse_args()

    logger.info(f"Args: runs={args.runs}, seed={args.seed}, "
                f"aug={args.aug}, no_aug={args.no_aug}, "
                f"use_optuna={args.use_optuna}")

    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")

    # ---------- 决定数据增强配置 ----------
    base_config = Config()
    if args.no_aug:
        augmentation_config = {k: False for k in base_config.augmentation_config}
    elif args.aug:
        augmentation_config = base_config.augmentation_config
    else:
        try:
            use_aug = input("是否使用数据增强？(y/n): ").lower() == 'y'
        except EOFError:
            use_aug = True
        augmentation_config = (base_config.augmentation_config if use_aug
                               else {k: False for k in base_config.augmentation_config})

    # ---------- Optuna 分支 ----------
    if args.use_optuna:
        if args.runs > 1:
            logger.warning("--use-optuna 与 --runs>1 不兼容，忽略 --runs")
        set_seed(args.seed)
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            direction='maximize', sampler=sampler,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=20)
        logger.info("Best trial:")
        logger.info(f"  Value:  {study.best_value}")
        logger.info(f"  Params: {study.best_params}")
        return

    # ---------- 普通训练（支持多次运行） ----------
    if args.runs > 1:
        run_multiple_experiments(args.runs, augmentation_config, base_seed=args.seed)
    else:
        results = run_single_training(augmentation_config, args.seed)
        logger.info(f"Final validation accuracy: {results['val_accuracy']:.2f}%")
        logger.info(f"Final sensitivity (macro): {results['sensitivity_macro']:.4f}")
        logger.info(f"Final specificity (macro): {results['specificity_macro']:.4f}")
        logger.info(f"Final F1-score (macro):    {results['f1_score_macro']:.4f}")


if __name__ == "__main__":
    logger.info("可用的数据增强选项:")
    default_controller = DataAugmentationController()
    for aug_name in default_controller.default_config.keys():
        logger.info(f"  - {aug_name}")

    main()
    logger.info("程序执行完成。")

# # 单次训练（保持旧行为）
# python your_script.py --runs 1 --aug
#
# # 5 次重复 + 输出均值/标准差
# python your_script.py --runs 5 --seed 42 --aug
#
# # 5 次重复，关闭数据增强
# python your_script.py --runs 5 --no-aug
#
# # 仍想用 Optuna
# python your_script.py --use-optuna --seed 42

# RUN 1/5  (seed=42)
# ...
# [Run 1/5] seed=42 -> val_acc = 57.14%, sens = 0.4645, spec = 0.9149, f1 = 0.4663
# RUN 2/5  (seed=43)
# ...
# [Run 2/5] seed=43 -> val_acc = 61.90%, sens = 0.5120, spec = 0.9280, f1 = 0.5103
# ...
#
# ======================================================================
# Aggregated results over 5 runs (base_seed=42):
# ----------------------------------------------------------------------
#   Run  1  (seed= 42): acc= 57.14%  sens=0.4645  spec=0.9149  f1=0.4663
#   Run  2  (seed= 43): acc= 61.90%  sens=0.5120  spec=0.9280  f1=0.5103
#   Run  3  (seed= 44): acc= 52.38%  sens=0.4100  spec=0.9000  f1=0.4110
#   Run  4  (seed= 45): acc= 59.52%  sens=0.4830  spec=0.9200  f1=0.4850
#   Run  5  (seed= 46): acc= 55.71%  sens=0.4520  spec=0.9100  f1=0.4550
# ----------------------------------------------------------------------
#   Accuracy    : 57.33% ± 3.55%   (95% CI ±3.11%)
#   Sensitivity : 0.4643 ± 0.0360
#   Specificity : 0.9146 ± 0.0104
#   F1-score    : 0.4655 ± 0.0357
# ======================================================================
# Summary saved to: ./models/runs_summary_20260102_194012.json