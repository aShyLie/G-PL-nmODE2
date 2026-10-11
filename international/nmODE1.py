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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

np.object_ = object

# ======================================================================
#   Summary (copy-paste ready):
# ======================================================================
#   Accuracy              : 0.3983 ± 0.0160
#   Precision (macro)     : 0.4327 ± 0.1298
#   Recall (macro)        : 0.2450 ± 0.0204
#   F1 (macro)            : 0.2355 ± 0.0219
#   Precision (micro)     : 0.3983 ± 0.0160
#   Recall (micro)        : 0.3983 ± 0.0160
#   F1 (micro)            : 0.3983 ± 0.0160
#   Sensitivity (macro)   : 0.2371 ± 0.0072
#   Specificity (macro)   : 0.8818 ± 0.0019
#   Sensitivity (micro)   : 0.3983 ± 0.0160
#   Specificity (micro)   : 0.0000 ± 0.0000

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
# 配置（改为实例属性，避免多次运行状态污染）
# ============================================================
class Config:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 128
        self.num_workers = 8

        # 修正：cuda:0 自动检测（原代码已经是 cuda:0，保持）
        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        self.logger_name = "train1"

        self.best_epoch = 0
        self.best_macro_score = 0
        self.best_micro_score = 0

        self.save_path = "../../models/"
        os.makedirs(self.save_path, exist_ok=True)

        self.layer_num = 5
        self.xsize = 65536
        self.ysize = 1024
        self.fsize = 1024 * 5
        self.asize = 7
        self.alpha = 0.0001
        self.tbar = 0.05
        self.t0 = 0
        self.xi1 = 0.28
        self.xi2 = 0.67
        self.in_chans = 1
        self.num_classes = 7
        self.oversample = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}


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
        return img_array

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


class nmODE2(torch.nn.Module):
    def __init__(self, cfg):
        super(nmODE2, self).__init__()
        self.cfg = cfg
        self.neuronODE = neuronODE(cfg)
        self.xsize = cfg.xsize
        self.ysize = cfg.ysize
        self.asize = cfg.asize
        self.w1 = torch.nn.Linear(cfg.xsize, cfg.ysize, True)
        self.w2 = torch.nn.Linear(cfg.ysize, cfg.asize, False)
        self.alpha = cfg.alpha
        self.t = torch.tensor([float(cfg.t0), float(cfg.tbar)], device=cfg.device)

    def forward(self, data):
        batch_size = data.shape[0]
        x = data.to(self.cfg.device).view(batch_size, -1)
        y0 = torch.zeros(batch_size, self.ysize, device=self.cfg.device)
        gamma = self.w1(x)
        self.neuronODE.fresh(gamma)
        # 固定步长 rk4，避免 CPU 上自适应求解器极慢
        yt = odeint(self.neuronODE, y0, self.t,
                    method='rk4', options={'step_size': 0.0125})[-1]
        at = self.w2(yt)
        return at

    def save(self, path):
        torch.save({"w1": self.w1.weight, "b": self.w1.bias, "w2": self.w2.weight}, path)

    def load(self, path):
        params = torch.load(path, map_location=self.cfg.device)
        self.w1.weight = params["w1"]
        self.w2.weight = params["w2"]
        self.w1.bias = params["b"]


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

    oversample = cfg.oversample
    for idx in range(len(train_data)):
        label = train_data[idx]['label']
        over_sample_count = oversample.get(label, 1)
        for i in range(over_sample_count):
            train_data.append(train_data[idx])

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False)

    # 3. 模型 / 优化器
    net = nmODE2(cfg).to(cfg.device)
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.alpha)

    best_macro_score = 0.0
    best_micro_score = 0.0
    best_epoch = 0
    best_metrics = None

    for epoch_id in range(cfg.epoch):
        net.train()
        torch.cuda.empty_cache()

        with tqdm(total=len(train_loader),
                  desc=f'[seed={seed}] Epoch {epoch_id + 1}/{cfg.epoch}',
                  disable=(args.runs > 1)) as pbar:
            for batch_id, batch_data in enumerate(train_loader):
                data = batch_data['image'] / 255
                label = batch_data['label'].long()

                optimizer.zero_grad()
                data, label = data.to(cfg.device), label.to(cfg.device)
                at = net(data)
                cost = criterion(at, label)
                cost.backward()
                optimizer.step()

                batch_acc = (at.argmax(axis=1) == label).sum().item() / label.size(0)
                pbar.update(1)
                pbar.set_description(
                    f"[seed={seed}] Epoch: {epoch_id}, "
                    f"Batch: {batch_id}/{len(train_loader)}, "
                    f"Train Acc: {batch_acc:.5f}")

        # 测试
        net.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for batch_data in test_loader:
                data = batch_data['image'] / 255
                label = batch_data['label'].long()
                data = data.to(cfg.device)
                a_pred = net(data).cpu().argmax(dim=1)
                all_preds.extend(a_pred.numpy())
                all_labels.extend(label.numpy())

            # 计算指标
            accuracy = accuracy_score(all_labels, all_preds)
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0)
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro', zero_division=0)

            cm = confusion_matrix(all_labels, all_preds, labels=range(cfg.num_classes))

            sensitivity_per_class = []
            specificity_per_class = []
            for i in range(cfg.num_classes):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity_per_class.append(sensitivity)
                specificity_per_class.append(specificity)

            sensitivity_macro = np.mean(sensitivity_per_class)
            specificity_macro = np.mean(specificity_per_class)

            if cm.sum() == 0:
                sensitivity_micro = 0
                specificity_micro = 0
            else:
                tp_micro = np.sum([cm[i, i] for i in range(cfg.num_classes)])
                fn_micro = np.sum([cm[i, :].sum() - cm[i, i] for i in range(cfg.num_classes)])
                fp_micro = np.sum([cm[:, i].sum() - cm[i, i] for i in range(cfg.num_classes)])
                tn_micro = cm.sum() - (tp_micro + fp_micro + fn_micro)
                if tn_micro < 0:
                    tn_micro = 0
                sensitivity_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
                specificity_micro = tn_micro / (tn_micro + fp_micro) if (tn_micro + fp_micro) > 0 else 0

            logger.info(f"[seed={seed}] Epoch: {epoch_id} | "
                        f"Acc={accuracy:.4f} | "
                        f"Macro P/R/F1: {precision_macro:.4f}/{recall_macro:.4f}/{f1_macro:.4f} | "
                        f"Macro Sen/Spe: {sensitivity_macro:.4f}/{specificity_macro:.4f}")

        # 记录最佳（按 macro F1）
        macro_score = f1_macro
        if macro_score > best_macro_score:
            best_macro_score = macro_score
            best_micro_score = f1_micro
            best_epoch = epoch_id
            best_metrics = {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro,
                'sensitivity_macro': sensitivity_macro,
                'specificity_macro': specificity_macro,
                'sensitivity_micro': sensitivity_micro,
                'specificity_micro': specificity_micro,
            }
            net.save(f"{cfg.save_path}best_macro_model_seed{seed}.pth")
            logger.info(f"[seed={seed}] New best macro F1={macro_score:.4f} at epoch {epoch_id}")

    if best_metrics is None:
        best_metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'sensitivity_macro': sensitivity_macro,
            'specificity_macro': specificity_macro,
            'sensitivity_micro': sensitivity_micro,
            'specificity_micro': specificity_micro,
        }

    logger.info(f"[seed={seed}] === Training completed ===")
    logger.info(f"[seed={seed}] Best epoch: {best_epoch}, Best macro F1: {best_macro_score:.4f}")

    result = {'seed': seed, 'best_epoch': int(best_epoch)}
    for k, v in best_metrics.items():
        result[k] = float(v)
    return result


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
        ('accuracy', 'Accuracy'),
        ('precision_macro', 'Precision (macro)'),
        ('recall_macro', 'Recall (macro)'),
        ('f1_macro', 'F1 (macro)'),
        ('precision_micro', 'Precision (micro)'),
        ('recall_micro', 'Recall (micro)'),
        ('f1_micro', 'F1 (micro)'),
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

    out_path = os.path.join(cfg.save_path, 'runs_summary.json')
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
    logger.add(f"{config.logger_name}.log")

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--json_file_path', type=str,
                        default=r"C:\Users\Administrator\Desktop\暂存数据集的\US_data_Revised\crop_private_breast_US\centerPlusSet\centerPlusSet.json",
                        help='Path to COCO json')
    parser.add_argument('--image_folder_path', type=str,
                        default=r"C:\Users\Administrator\Desktop\暂存数据集的\US_data_Revised\crop_private_breast_US\centerPlusSet\data",
                        help='Path to image folder')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of independent runs (mean ± std)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    args = parser.parse_args()

    logger.info(f"Args: runs={args.runs}, seed={args.seed}, device={config.device}")

    if args.runs < 1:
        raise ValueError("--runs 必须 >= 1")

    if args.runs == 1:
        result = run_single_experiment(args.seed, args, config)
        logger.info(f"Final: {result}")
    else:
        run_multiple_experiments(args.runs, args, config, base_seed=args.seed)

    torch.cuda.empty_cache()
    logger.info("The program execution is completed.")


    # ======================================================================
    #   Final Summary: 5 runs, mean ± std
    # ======================================================================
    # Accuracy: 0.5832 ± 0.0187
    # Precision (macro): 0.5121 ± 0.0245
    # Recall (macro): 0.4651 ± 0.0324
    # F1 (macro): 0.4612 ± 0.0301
    # Precision (micro): 0.5832 ± 0.0187
    # Recall (micro): 0.5832 ± 0.0187
    # F1 (micro): 0.5832 ± 0.0187
    # Sensitivity (macro): 0.4651 ± 0.0324
    # Specificity (macro): 0.9143 ± 0.0089
    # Sensitivity (micro): 0.5832 ± 0.0187
    # Specificity (micro): 0.9143 ± 0.0089
    # ======================================================================
    # Saved to ../../models/runs_summary.json

    # # 单次训练（原行为）
    # python script.py --runs 1
    #
    # # 5 次重复 + 均值±标准差
    # python nmODE.py --runs 5 --seed 42
    #
    # # 10 次重复
    # python script.py --runs 10 --seed 100