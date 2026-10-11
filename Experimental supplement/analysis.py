"""
analysis.py
在测试集上计算：
1. 收敛残差 ||dp/dt|| 的均值 + 95% CI
2. 模态解耦指标：互信息、线性探测、CKA、跨模态检索
3. 显著性图（输入梯度）
4. 所有指标的 bootstrap 95% CI

用法：
    python analysis.py --data_root D:\gap\gap-data\BrEaST \
                       --ckpt ../models/best_breast_cancer_model.pth \
                       --out_dir ./analysis_results --n_boot 1000
"""
import os
import argparse
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import PCA
from torchdiffeq import odeint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# 明确导入需要的符号
# ============================================================
from G_PL_BrEaST import (
    Config,
    BreastCancerDataset,
    create_data_loaders,
    ImageProcessor,
    neuronODE,
    nmODE2,
)

config = Config()


# ============================================================
# 0. bootstrap 95% CI
# ============================================================
def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    mean = values.mean()
    std = values.std(ddof=1) if n > 1 else 0.0
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(values[idx].mean())
    boots = np.sort(boots)
    lo = boots[int(n_boot * alpha / 2)]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return float(mean), float(std), float(lo), float(hi)


# ============================================================
# 1. 收敛残差
# ============================================================
@torch.no_grad()
def compute_residual(net, loader, image_processor, device):
    net.eval()
    per_sample = []

    for batch_data in tqdm(loader, desc="[residual]"):
        image1 = batch_data['image1']
        image2 = batch_data['image2']

        images = torch.cat([image1, image2], dim=1) * 255.0
        processed = image_processor(images).to(device)

        B = processed.shape[0]
        x = processed.view(B, -1)
        gamma = net.w1(x)
        net.neuronODE.fresh(gamma)

        y0 = torch.zeros(B, net.ysize, device=device)
        yt = odeint(net.neuronODE, y0, net.t)[-1]
        dpdt = net.neuronODE(net.t[-1], yt)
        residual_per_sample = dpdt.norm(dim=1).cpu().numpy()
        per_sample.append(residual_per_sample)

    per_sample = np.concatenate(per_sample, axis=0)
    return per_sample


# ============================================================
# 2. 模态解耦指标
# ============================================================
@torch.no_grad()
def extract_features(net, loader, image_processor, device):
    net.eval()
    z_img, z_txt, z_label = [], [], []

    for batch_data in tqdm(loader, desc="[features]"):
        image1 = batch_data['image1']
        image2 = batch_data['image2']
        label = batch_data['label'].long()

        images = torch.cat([image1, image2], dim=1) * 255.0
        processed = image_processor(images).to(device)

        B = processed.shape[0]
        x = processed.view(B, -1)
        gamma = net.w1(x)
        net.neuronODE.fresh(gamma)

        y0 = torch.zeros(B, net.ysize, device=device)
        yt = odeint(net.neuronODE, y0, net.t)[-1]

        img_feat = yt.cpu().numpy()
        if hasattr(net, 'text_encoder') and 'text' in batch_data:
            text_tensor = batch_data['text']
            text_feat = net.text_encoder(text_tensor.to(device)).cpu().numpy()
        else:
            # BrEaST 单模态：用 gamma 作为"占位文本特征"
            text_feat = gamma.cpu().numpy()

        z_img.append(img_feat)
        z_txt.append(text_feat)
        z_label.append(label.numpy())

    z_img = np.concatenate(z_img, axis=0)
    z_txt = np.concatenate(z_txt, axis=0)
    z_label = np.concatenate(z_label, axis=0)
    return {'z_img': z_img, 'z_txt': z_txt, 'label': z_label}


# ============================================================
# quantize_for_mi：位打包，避免组合爆炸 + int64 溢出
# ============================================================
def quantize_for_mi(x, n_bins=8, max_dim=4):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]

    d = x.shape[1]
    if d > max_dim:
        n_comp = min(max_dim, d, x.shape[0])
        if n_comp >= 2:
            pca = PCA(n_components=n_comp)
            x = pca.fit_transform(x)
        else:
            x = x[:, :n_comp]

    n_samples, n_features = x.shape
    codes = np.zeros((n_samples, n_features), dtype=np.int64)
    for j in range(n_features):
        col = x[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-12:
            codes[:, j] = 0
        else:
            edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
            codes[:, j] = np.clip(np.digitize(col, edges) - 1, 0, n_bins - 1)

    bits_per_dim = max(1, int(np.ceil(np.log2(max(n_bins, 2)))))
    combined = np.zeros(n_samples, dtype=np.int64)
    for j in range(n_features):
        mask = (1 << bits_per_dim) - 1
        combined |= (codes[:, j] & mask) << (bits_per_dim * j)

    return combined


def norm_mi(mi_val, a, b):
    """归一化互信息 NMI = MI / min(H(a), H(b))"""
    _, pa_counts = np.unique(np.asarray(a), return_counts=True)
    _, pb_counts = np.unique(np.asarray(b), return_counts=True)

    pa = pa_counts.astype(np.float64) / max(pa_counts.sum(), 1)
    pb = pb_counts.astype(np.float64) / max(pb_counts.sum(), 1)

    ha = entropy(pa)
    hb = entropy(pb)
    denom = min(ha, hb)
    return float(mi_val / denom) if denom > 1e-12 else 0.0


def mutual_information(img_feat, txt_feat, label, n_bins=8):
    img_disc = quantize_for_mi(img_feat, n_bins)
    txt_disc = quantize_for_mi(txt_feat, n_bins)
    lab_disc = np.abs(label.astype(np.int64))

    img_disc = np.abs(img_disc)
    txt_disc = np.abs(txt_disc)

    mi_img_text = mutual_info_score(img_disc, txt_disc)
    mi_img_lab = mutual_info_score(img_disc, lab_disc)
    mi_txt_lab = mutual_info_score(txt_disc, lab_disc)

    return {
        'MI_img_text': float(mi_img_text),
        'MI_img_label': float(mi_img_lab),
        'MI_text_label': float(mi_txt_lab),
        'NMI_img_text': norm_mi(mi_img_text, img_disc, txt_disc),
        'NMI_img_label': norm_mi(mi_img_lab, img_disc, lab_disc),
        'NMI_text_label': norm_mi(mi_txt_lab, txt_disc, lab_disc),
    }


def linear_probe(img_feat, txt_feat, label, cv=5):
    clf_img = LogisticRegression(max_iter=2000)
    clf_txt = LogisticRegression(max_iter=2000)

    n_classes = len(np.unique(label))
    cv_img = min(cv, n_classes) if n_classes > 1 else 2
    cv_txt = cv_img

    try:
        acc_img = cross_val_score(clf_img, img_feat, label, cv=cv_img,
                                  scoring='accuracy').mean()
    except Exception as e:
        print(f"[warn] linear_probe img failed: {e}")
        acc_img = float('nan')

    try:
        acc_txt = cross_val_score(clf_txt, txt_feat, label, cv=cv_txt,
                                  scoring='accuracy').mean()
    except Exception as e:
        print(f"[warn] linear_probe txt failed: {e}")
        acc_txt = float('nan')

    return {
        'linear_probe_img_acc': float(acc_img),
        'linear_probe_txt_acc': float(acc_txt),
    }


def cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    XTX = X.T @ X
    YTY = Y.T @ Y
    XTY = X.T @ Y
    num = (XTY ** 2).sum()
    den = np.linalg.norm(XTX) * np.linalg.norm(YTY)
    return float(num / (den + 1e-12))


def cross_modal_retrieval(img_feat, txt_feat, label, topk=(1, 5)):
    img_n = F.normalize(torch.tensor(img_feat, dtype=torch.float32), dim=1)
    txt_n = F.normalize(torch.tensor(txt_feat, dtype=torch.float32), dim=1)

    sim = img_n @ txt_n.T
    rank = sim.argsort(dim=1, descending=True)

    label = torch.tensor(label)
    same_label = (label[:, None] == label[None, :])
    same_label.fill_diagonal_(False)

    results = {}
    for k in topk:
        kk = min(k, img_n.shape[0] - 1)
        topk_idx = rank[:, :kk]
        hit = torch.gather(same_label, 1, topk_idx)
        results[f'recall@{k}_img2txt'] = float(hit.any(dim=1).float().mean().item())

    sim_t = txt_n @ img_n.T
    rank_t = sim_t.argsort(dim=1, descending=True)
    for k in topk:
        kk = min(k, txt_n.shape[0] - 1)
        topk_idx = rank_t[:, :kk]
        hit = torch.gather(same_label, 1, topk_idx)
        results[f'recall@{k}_txt2img'] = float(hit.any(dim=1).float().mean().item())

    return results


# ============================================================
# 3. 显著性图（★ 修复：必须拼接 image1 + image2）
# ============================================================
def saliency_map(net, image_processor, img1_batch, img2_batch, device):
    """
    对 image1 求输入梯度；image2 作为常量一起送入网络。
    返回 sal (B, H, W) 和 pred (B,)。
    """
    net.eval()
    net.zero_grad()

    img1 = img1_batch.clone().detach().to(device) * 255.0
    img1.requires_grad_(True)
    img2 = (img2_batch.clone().detach().to(device) * 255.0)

    images = torch.cat([img1, img2], dim=1)   # ★ 拼接：15 + 15 = 30 通道
    processed = image_processor(images)

    logits = net(processed)
    pred = logits.argmax(dim=1)

    grad = torch.autograd.grad(
        outputs=logits.gather(1, pred.unsqueeze(1)).sum(),
        inputs=img1,                          # ★ 只对 image1 求梯度
        retain_graph=False,
        create_graph=False,
    )[0]

    sal = grad.abs().mean(dim=1).detach().cpu().numpy()   # (B, H, W)
    return sal, pred.detach().cpu().numpy()


def _to_display(img_chw):
    """把 (C,H,W) 张量转为可 imshow 的 (H,W,3) 或 (H,W)。"""
    img = np.asarray(img_chw, dtype=np.float32)
    if img.ndim == 2:
        vis = img
    elif img.shape[0] >= 3:
        vis = np.transpose(img[:3], (1, 2, 0))
    else:
        vis = img[0]
    vmin, vmax = vis.min(), vis.max()
    if vmax - vmin > 1e-8:
        vis = (vis - vmin) / (vmax - vmin)
    return vis


# ============================================================
# 4. 主流程
# ============================================================
def run_analysis():
    parser = argparse.ArgumentParser(description='Quantitative Analysis for nmODE')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='../models/best_breast_cancer_model.pth',
                        help='训练好的 best model 路径（不存在会自动搜索）')
    parser.add_argument('--out_dir', type=str, default='./analysis_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_bins', type=int, default=8,
                        help='MI 离散化 bins（默认 8，小样本请勿设太大）')
    parser.add_argument('--n_boot', type=int, default=1000, help='bootstrap 次数')
    parser.add_argument('--n_vis', type=int, default=8, help='显著性图数量')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = config.device

    # ---- 数据 ----
    train_loader, test_loader = create_data_loaders(args.data_root, args.batch_size)

    # ---- 模型 ----
    image_processor = ImageProcessor(alpha=config.img_alpha).to(device)
    net = nmODE2().to(device)

    # 自动搜索模型文件
    if os.path.exists(args.ckpt):
        ckpt_path = args.ckpt
    else:
        search_dirs = []
        d = os.path.dirname(args.ckpt)
        if d:
            search_dirs.append(d)
        search_dirs += ['./models', '../models', '.']
        seen = set()
        search_dirs = [x for x in search_dirs if not (x in seen or seen.add(x))]

        candidates = []
        for d in search_dirs:
            if os.path.isdir(d):
                candidates.extend(sorted(glob.glob(
                    os.path.join(d, 'best_breast_cancer_model*.pth'))))
        if not candidates:
            raise FileNotFoundError(
                f"找不到模型文件。请先运行 train_alpha.py 生成模型，"
                f"或显式指定 --ckpt 路径。\n"
                f"传入路径: {args.ckpt}\n"
                f"搜索目录: {search_dirs}\n"
                f"提示: dir ..\\models\\best_*.pth")
        ckpt_path = candidates[0]
        print(f"[Info] 未找到 {args.ckpt}，自动使用 {ckpt_path}")

    net.load(ckpt_path)
    net.eval()
    print(f"[Info] 已加载模型: {ckpt_path}")

    # ============================================================
    # (1) 收敛残差
    # ============================================================
    print("\n" + "=" * 60)
    print("(1) 收敛残差 ||dp/dt||")
    print("=" * 60)

    residual_per_sample = compute_residual(net, test_loader, image_processor, device)
    mean, std, lo, hi = bootstrap_ci(residual_per_sample, n_boot=args.n_boot)
    print(f"  均值: {mean:.6f}")
    print(f"  标准差: {std:.6f}")
    print(f"  95% CI: [{lo:.6f}, {hi:.6f}]")
    print(f"  中位数: {np.median(residual_per_sample):.6f}")
    print(f"  最大值: {np.max(residual_per_sample):.6f}")
    print(f"  样本数: {len(residual_per_sample)}")

    residual_result = {
        'mean': mean, 'std': std, 'ci_low': lo, 'ci_high': hi,
        'median': float(np.median(residual_per_sample)),
        'max': float(np.max(residual_per_sample)),
        'min': float(np.min(residual_per_sample)),
        'n': int(len(residual_per_sample)),
    }

    plt.figure(figsize=(7, 4))
    plt.hist(residual_per_sample, bins=50, color='steelblue', alpha=0.8)
    plt.axvline(mean, color='red', linestyle='--', label=f'mean={mean:.4f}')
    plt.xlabel('||dp/dt|| at t=tbar')
    plt.ylabel('count')
    plt.title('Convergence Residual Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'residual_hist.png'), dpi=150)
    plt.close()

    # ============================================================
    # (2) 模态解耦
    # ============================================================
    print("\n" + "=" * 60)
    print("(2) 模态解耦指标")
    print("=" * 60)

    feats = extract_features(net, test_loader, image_processor, device)
    z_img, z_txt, z_lab = feats['z_img'], feats['z_txt'], feats['label']

    mi_result = mutual_information(z_img, z_txt, z_lab, n_bins=args.n_bins)
    print("\n  [互信息]")
    for k, v in mi_result.items():
        print(f"    {k}: {v:.6f}")

    lp_result = linear_probe(z_img, z_txt, z_lab, cv=5)
    print("\n  [线性探测]")
    for k, v in lp_result.items():
        print(f"    {k}: {v:.6f}")

    cka_val = cka(z_img, z_txt)
    print(f"\n  [CKA] img vs text: {cka_val:.6f}")

    retrieval_result = cross_modal_retrieval(z_img, z_txt, z_lab, topk=(1, 5))
    print("\n  [跨模态检索]")
    for k, v in retrieval_result.items():
        print(f"    {k}: {v:.6f}")

    decouple_result = {**mi_result, **lp_result,
                       'cka_img_text': cka_val, **retrieval_result}

    # ============================================================
    # (3) 显著性图
    # ============================================================
    print("\n" + "=" * 60)
    print("(3) 显著性图")
    print("=" * 60)

    sal_images = []
    count = 0
    for batch_data in test_loader:
        image1 = batch_data['image1']
        image2 = batch_data['image2']               # ★ 取 image2
        sal, pred = saliency_map(net, image_processor, image1, image2, device)
        for i in range(sal.shape[0]):
            sal_images.append((image1[i].numpy(), sal[i], pred[i]))
            count += 1
            if count >= args.n_vis:
                break
        if count >= args.n_vis:
            break

    n_vis = len(sal_images)
    if n_vis > 0:
        fig, axes = plt.subplots(2, n_vis, figsize=(2.2 * n_vis, 5))
        if n_vis == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for i in range(n_vis):
            img, sal, pred = sal_images[i]
            vis = _to_display(img)                  # ★ 支持 15 通道
            axes[0, i].imshow(vis)
            axes[0, i].set_title(f'pred={pred}', fontsize=9)
            axes[0, i].axis('off')
            axes[1, i].imshow(vis)
            axes[1, i].imshow(sal, cmap='jet', alpha=0.5)
            axes[1, i].set_title('saliency', fontsize=9)
            axes[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'saliency_maps.png'), dpi=150)
        plt.close()
        print(f"  已保存 {n_vis} 张显著性图")

    # ============================================================
    # (4) 汇总
    # ============================================================
    all_results = {
        'residual': residual_result,
        'decoupling': decouple_result,
        'config': {
            'xi1': config.xi1,
            'xi2': config.xi2,
            'xi1_plus_xi2': config.xi1 + config.xi2,
            'tbar': config.tbar,
            'ysize': config.ysize,
            'image_size': config.image_size,
            'n_test': int(len(z_lab)),
            'ckpt_path': ckpt_path,
            'n_bins': int(args.n_bins),
        }
    }
    out_json = os.path.join(args.out_dir, 'quantitative_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] 全部结果已保存: {out_json}")

    # ============================================================
    # (5) LaTeX 表
    # ============================================================
    tex_lines = []
    tex_lines.append("% ===== Table X: Convergence Residual =====")
    tex_lines.append("\\begin{table}[h]\\centering")
    tex_lines.append("\\caption{Convergence residual $\\|dp/dt\\|$ on the test set.}")
    tex_lines.append("\\begin{tabular}{lc}\\hline")
    tex_lines.append("Metric & Value \\\\ \\hline")
    tex_lines.append("Mean $\\pm$ std & "
                     + f"${residual_result['mean']:.4f} \\pm {residual_result['std']:.4f}$"
                     + " \\\\")
    tex_lines.append("95\\% CI & "
                     + f"$[{residual_result['ci_low']:.4f},\\ {residual_result['ci_high']:.4f}]$"
                     + " \\\\")
    tex_lines.append("Median & "
                     + f"${residual_result['median']:.4f}$" + " \\\\")
    tex_lines.append("Max & "
                     + f"${residual_result['max']:.4f}$" + " \\\\")
    tex_lines.append("$N$ & " + str(residual_result['n']) + " \\\\")
    tex_lines.append("\\hline\\end{tabular}\\end{table}")
    tex_lines.append("")
    tex_lines.append("% ===== Table Y: Modality Decoupling =====")
    tex_lines.append("\\begin{table}[h]\\centering")
    tex_lines.append("\\caption{Modality decoupling metrics on the test set.}")
    tex_lines.append("\\begin{tabular}{lc}\\hline")
    tex_lines.append("Metric & Value \\\\ \\hline")
    for k, v in decouple_result.items():
        tex_lines.append(k + " & " + f"${v:.4f}$" + " \\\\")
    tex_lines.append("\\hline\\end{tabular}\\end{table}")

    with open(os.path.join(args.out_dir, 'tables.tex'), 'w', encoding='utf-8') as f:
        f.write("\n".join(tex_lines))
    print(f"[Saved] LaTeX 表: {os.path.join(args.out_dir, 'tables.tex')}")


if __name__ == "__main__":
    run_analysis()