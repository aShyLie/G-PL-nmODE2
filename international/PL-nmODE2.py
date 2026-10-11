import argparse
from tqdm import tqdm
from loguru import logger
import torch
from torchdiffeq import odeint
import json
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
from sklearn.metrics import confusion_matrix
from torchvision import transforms

# ======================================================================
#   Summary (copy-paste ready):
# ======================================================================
#   Accuracy              : 0.6073 ± 0.0205
#   Sensitivity (macro)   : 0.3650 ± 0.0093
#   Specificity (macro)   : 0.9201 ± 0.0036
#   F1-score (macro)      : 0.3850 ± 0.0132
#   Sensitivity (micro)   : 0.6073 ± 0.0205
#   Specificity (micro)   : 0.9345 ± 0.0034
#   F1-score (micro)      : 0.6073 ± 0.0205


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
# 配置
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 50
        self.batch_size = 128
        self.num_workers = 8

        # 修改：cuda:1 → cuda:0（自动检测），单卡机器上不会 fallback 到 CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.logger_name = "train"
        self.best_epoch = 0
        self.best_accuracy = 0
        self.best_f1_macro = 0
        self.best_sensitivity_macro = 0
        self.save_path = "../models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.layer_num = 5
        self.xsize = 327680
        self.ysize = 2048
        self.fsize = 2048 * 5
        self.asize = 10
        self.alpha = 0.00001
        self.tbar = 0.1
        self.t0 = 0
        self.xi1 = 0.6
        self.xi2 = 0.6
        self.in_chans = 1
        self.num_classes = 7


config = Config()


# ============================================================
# 数据类
# ============================================================
class JunQuZongData:
    def __init__(self, json_file_path, image_folder_path):
        self.json_file_path = json_file_path
        self.image_folder_path = image_folder_path
        self.image_category_map = {}
        self.categories = {}

    def load_data(self):
        with open(self.json_file_path, 'r') as f:
            coco_data = json.load(f)

        self.categories = {category['id']: category['name'] for category in coco_data['categories']}

        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            image_filename = coco_data['images'][image_id]['file_name'].split('/')[-1]
            self.image_category_map[image_filename] = category_id

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(256, 256))
        img = img.convert('L')
        img_array = img_to_array(img)
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)
        return img_tensor

    def create_dataset(self, X_data, y_data):
        dataset = []
        for i in range(len(X_data)):
            filename = X_data[i]
            category_id = y_data[i]
            image_filepath = os.path.join(self.image_folder_path, filename)
            if os.path.exists(image_filepath) and filename in self.image_category_map:
                X = self.preprocess_image(image_filepath)
                y = category_id
                dataset.append({'image': X, 'label': y})
        return dataset


# ============================================================
# ImageProcessor
# ============================================================
class ImageProcessor(torch.nn.Module):
    def __init__(self):
        super(ImageProcessor, self).__init__()
        self.layer_num = config.layer_num

    def forward(self, data):
        transformed_images = torch.zeros(
            size=(data.shape[0], self.layer_num, data.shape[2], data.shape[3]),
            dtype=torch.float32, device=data.device)

        pixel = data[:, 0, :, :]
        s1 = 1 - pixel / 64.0
        s2 = 1 - s1
        s3 = 1 - (pixel - 63) / 64.0
        s4 = 1 - s3
        s5 = 1 - (pixel - 127) / 64.0
        s6 = 1 - s5
        s7 = 1 - (pixel - 191) / 64.0
        s8 = 1 - s7

        transformed_images[:, 0] = torch.where((0 <= pixel) & (pixel <= 63), s1, transformed_images[:, 0])
        transformed_images[:, 1] = torch.where((0 <= pixel) & (pixel <= 63), s2, transformed_images[:, 1])
        transformed_images[:, 1] = torch.where((64 <= pixel) & (pixel <= 127), s3, transformed_images[:, 1])
        transformed_images[:, 2] = torch.where((64 <= pixel) & (pixel <= 127), s4, transformed_images[:, 2])
        transformed_images[:, 2] = torch.where((128 <= pixel) & (pixel <= 191), s5, transformed_images[:, 2])
        transformed_images[:, 3] = torch.where((128 <= pixel) & (pixel <= 191), s6, transformed_images[:, 3])
        transformed_images[:, 3] = torch.where((192 <= pixel) & (pixel <= 255), s7, transformed_images[:, 3])
        transformed_images[:, 4] = torch.where((192 <= pixel) & (pixel <= 255), s8, transformed_images[:, 4])
        return transformed_images


# ============================================================
# ODE 模型
# ============================================================
class neuronODE(torch.nn.Module):
    def __init__(self):
        super(neuronODE, self).__init__()
        self.gamma = None

    def fresh(self, gamma):
        self.gamma = torch.clamp(gamma, -5.0, 5.0)

    def forward(self, t, p):
        dpdt = -p + torch.pow(
            torch.sin(config.xi1 * p + torch.cos(config.xi2 * p + self.gamma)), 2)
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
        # 固定步长 rk4，避免 CPU 上自适应求解器极慢
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.05})[-1]
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
# 指标计算
# ============================================================
def calculate_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (TP + FP + FN)

    accuracy = np.sum(TP) / np.sum(cm) if np.sum(cm) > 0 else 0

    sensitivity = np.divide(TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) != 0)
    specificity = np.divide(TN, TN + FP, out=np.zeros_like(TN, dtype=float), where=(TN + FP) != 0)
    f1_score = np.divide(2 * TP, 2 * TP + FP + FN, out=np.zeros_like(TP, dtype=float),
                         where=(2 * TP + FP + FN) != 0)

    sensitivity_macro = np.mean(sensitivity)
    specificity_macro = np.mean(specificity)
    f1_score_macro = np.mean(f1_score)

    denom_sen = np.sum(TP + FN)
    denom_spe = np.sum(TN + FP)
    denom_f1 = 2 * np.sum(TP) + np.sum(FP) + np.sum(FN)

    sensitivity_micro = np.sum(TP) / denom_sen if denom_sen > 0 else 0
    specificity_micro = np.sum(TN) / denom_spe if denom_spe > 0 else 0
    f1_score_micro = 2 * np.sum(TP) / denom_f1 if denom_f1 > 0 else 0

    return (accuracy, sensitivity_macro, specificity_macro, f1_score_macro,
            sensitivity_micro, specificity_micro, f1_score_micro)


# ============================================================
# 单次完整实验
# ============================================================
def run_single_experiment(seed, args):
    """用指定 seed 跑一次完整训练，返回最终测试指标。"""
    set_seed(seed)

    logger.info(f"\n{'='*70}")
    logger.info(f"RUN (seed={seed})")
    logger.info(f"{'='*70}")

    # 1. 加载数据
    data_loader = JunQuZongData(
        json_file_path=args.json_file_path,
        image_folder_path=args.image_folder_path)
    data_loader.load_data()

    X_data = list(data_loader.image_category_map.keys())
    y_data = list(data_loader.image_category_map.values())
    dataset = data_loader.create_dataset(X_data, y_data)

    # 2. 划分数据集（★ 随 seed 变化）
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=seed)

    oversample = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    for idx in range(len(train_data)):
        label = train_data[idx]['label']
        over_sample_count = oversample.get(label, 1)
        for i in range(over_sample_count):
            train_data.append(train_data[idx])

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    # 3. 模型 / 优化器
    net = nmODE2().to(config.device)
    criterion = torch.nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.alpha)

    best_metrics = {
        'accuracy': 0,
        'sensitivity_macro': 0,
        'specificity_macro': 0,
        'f1_score_macro': 0,
        'epoch': 0,
        'sensitivity_micro': 0,
        'specificity_micro': 0,
        'f1_score_micro': 0,
    }

    # 训练过程不再重复保存中间模型（多 run 时开销大），只记录指标
    for epoch_id in range(config.epoch):
        net.train()
        torch.cuda.empty_cache()

        with tqdm(total=len(train_loader),
                  desc=f'[seed={seed}] Epoch {epoch_id + 1}/{config.epoch}',
                  disable=(args.runs > 1)) as pbar:
            for batch_id, batch_data in enumerate(train_loader):
                data = batch_data['image']
                label = batch_data['label'].long()

                data = data.view(data.shape[0], 1, 256, 256)
                process = ImageProcessor()
                data = process.forward(data)

                optimizer.zero_grad()
                data, label = data.to(config.device), label.to(config.device)
                at = net(data)
                cost = criterion(at, label)
                cost.backward()
                optimizer.step()

                batch_acc = (at.argmax(axis=1) == label).sum() / label.shape[0]
                pbar.update(1)
                pbar.set_description(
                    f"[seed={seed}] Epoch: {epoch_id}, "
                    f"Batch: {batch_id}/{len(train_loader)}, "
                    f"Train Acc: {batch_acc:.5f}")

        # 测试
        net.eval()
        with torch.no_grad():
            y_true_all, y_pred_all = [], []
            for batch_id, batch_data in enumerate(test_loader):
                data = batch_data['image']
                label = batch_data['label'].long()
                data = data.view(data.shape[0], 1, 256, 256)
                process = ImageProcessor()
                data = process.forward(data)
                data = data.to(config.device)
                a_pred = net(data).cpu().argmax(dim=1)
                y_true_all.extend(label.numpy())
                y_pred_all.extend(a_pred.numpy())

            (accuracy, sensitivity_macro, specificity_macro, f1_score_macro,
             sensitivity_micro, specificity_micro, f1_score_micro) = calculate_metrics(
                y_true_all, y_pred_all, config.num_classes)

            logger.info(f"[seed={seed}] Epoch: {epoch_id} | "
                        f"Acc={accuracy:.4f} | "
                        f"Sen_macro={sensitivity_macro:.4f} | "
                        f"Spe_macro={specificity_macro:.4f} | "
                        f"F1_macro={f1_score_macro:.4f}")

        # 更新最佳指标（按 macro F1）
        if f1_score_macro > best_metrics['f1_score_macro']:
            best_metrics['f1_score_macro'] = f1_score_macro
            best_metrics['accuracy'] = accuracy
            best_metrics['sensitivity_macro'] = sensitivity_macro
            best_metrics['specificity_macro'] = specificity_macro
            best_metrics['sensitivity_micro'] = sensitivity_micro
            best_metrics['specificity_micro'] = specificity_micro
            best_metrics['f1_score_micro'] = f1_score_micro
            best_metrics['epoch'] = epoch_id

            # 每个 run 单独保存一份最佳模型
            torch.save(net.state_dict(),
                       os.path.join(config.save_path, f"best_f1_model_seed{seed}.pth"))
            logger.info(f"[seed={seed}] Best model saved at Epoch {epoch_id} "
                        f"with f1_macro={f1_score_macro:.4f}")

    return {
        'seed': seed,
        'best_epoch': int(best_metrics['epoch']),
        'accuracy': float(best_metrics['accuracy']),
        'sensitivity_macro': float(best_metrics['sensitivity_macro']),
        'specificity_macro': float(best_metrics['specificity_macro']),
        'f1_score_macro': float(best_metrics['f1_score_macro']),
        'sensitivity_micro': float(best_metrics['sensitivity_micro']),
        'specificity_micro': float(best_metrics['specificity_micro']),
        'f1_score_micro': float(best_metrics['f1_score_micro']),
    }


# ============================================================
# 多次重复实验
# ============================================================
def run_multiple_experiments(runs, args, base_seed=42):
    all_results = []
    for i in range(runs):
        seed = base_seed + i
        try:
            result = run_single_experiment(seed, args)
            all_results.append(result)
            logger.info(f"[Run {i + 1}/{runs}] seed={seed} -> "
                        f"Acc={result['accuracy']:.4f}, "
                        f"F1_macro={result['f1_score_macro']:.4f}")
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
        ('sensitivity_macro', 'Sensitivity (macro)'),
        ('specificity_macro', 'Specificity (macro)'),
        ('f1_score_macro', 'F1-score (macro)'),
        ('sensitivity_micro', 'Sensitivity (micro)'),
        ('specificity_micro', 'Specificity (micro)'),
        ('f1_score_micro', 'F1-score (micro)'),
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
    out_path = os.path.join(config.save_path, 'runs_summary.json')
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nmODE on breast cancer dataset')
    parser.add_argument('--json_file_path', type=str,
                        default=r"C:\Users\Administrator\Desktop\暂存数据集的\US_data_Revised\crop_private_breast_US\JunQuZongData\JunQuZongData.json",
                        help='Path to COCO json')
    parser.add_argument('--image_folder_path', type=str,
                        default=r"C:\Users\Administrator\Desktop\暂存数据集的\US_data_Revised\crop_private_breast_US\JunQuZongData\data",
                        help='Path to image folder')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent runs (mean ± std)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    args = parser.parse_args()

    logger.add(f"{config.logger_name}.log")
    logger.info(f"Args: runs={args.runs}, seed={args.seed}, device={config.device}")

    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")

    if args.runs == 1:
        result = run_single_experiment(args.seed, args)
        logger.info(f"Final: {result}")
    else:
        run_multiple_experiments(args.runs, args, base_seed=args.seed)

    torch.cuda.empty_cache()
    logger.info("The program execution is completed.")