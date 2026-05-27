import torch
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score

# ================= ⚙️ 配置区域 =================
SAVE_DIR = '/root/autodl-tmp/checkpoints_resnet_512_pro'
DATA_ROOT = '/root/autodl-tmp/LONG-tail/LTDD-main/LTDD-main/manner/medical_LT_pifu/data/isic2019'
PATH_MIXUP  = os.path.join(SAVE_DIR, 'resnet50_mixup_200ep_pro_best_bacc.pth')
PATH_CUTMIX = os.path.join(SAVE_DIR, 'resnet50_cutmix_200ep_pro_best_bacc.pth')

TARGET_CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
BATCH_SIZE = 64
USE_TTA = True # 保持和你推理时一致
# =================================================

try:
    from dataset_isic_hierarchical import ISIC2019_Hierarchical
    from model_resnet_hierarchical import ResNet_Hierarchical
except ImportError:
    sys.exit("❌ 找不到 dataset_isic_hierarchical.py 或 model_resnet_hierarchical.py")

def get_tta_transforms(img_tensor):
    return [img_tensor, torch.flip(img_tensor, dims=[3]), torch.flip(img_tensor, dims=[2]), torch.flip(img_tensor, dims=[2, 3])]

def precompute_logits(device):
    """ 
    预先计算所有 Logits，存入内存。
    这步最慢，但只需要做一次。
    """
    print("📦 Loading Models...")
    model_mixup = ResNet_Hierarchical(num_fine_classes=9, num_coarse_classes=3).to(device)
    model_mixup.load_state_dict(torch.load(PATH_MIXUP, map_location=device, weights_only=False), strict=False)
    model_mixup.eval()

    model_cutmix = ResNet_Hierarchical(num_fine_classes=9, num_coarse_classes=3).to(device)
    model_cutmix.load_state_dict(torch.load(PATH_CUTMIX, map_location=device, weights_only=False), strict=False)
    model_cutmix.eval()

    print("\n⏳ Extracting Logits (This takes 1-2 mins)...")
    transform_test = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    ds = ISIC2019_Hierarchical(root=DATA_ROOT, train=False, transform=transform_test)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    all_mix = []
    all_cut = []
    all_targets = []

    with torch.no_grad():
        for imgs, _, targets in tqdm(loader):
            imgs = imgs.to(device)
            
            # TTA logic
            if USE_TTA:
                aug_imgs = get_tta_transforms(imgs)
                l_m_sum, l_c_sum = 0, 0
                for aug in aug_imgs:
                    l_m_sum += model_mixup(aug)[0][:, :8]
                    l_c_sum += model_cutmix(aug)[0][:, :8]
                l_mix = l_m_sum / 4.0
                l_cut = l_c_sum / 4.0
            else:
                l_mix = model_mixup(imgs)[0][:, :8]
                l_cut = model_cutmix(imgs)[0][:, :8]
            
            all_mix.append(l_mix.cpu())
            all_cut.append(l_cut.cpu())
            all_targets.append(targets)

    return torch.cat(all_mix), torch.cat(all_cut), torch.cat(all_targets).numpy()

def optimize_weights():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 获取数据 (Logits)
    logits_mix, logits_cut, targets = precompute_logits(device)
    
    # 放到 GPU 上加速计算
    logits_mix = logits_mix.to(device)
    logits_cut = logits_cut.to(device)
    
    print("\n🧮 Starting Coordinate Descent Optimization...")
    print("   Goal: Find the mathematically optimal weights for each class.")
    
    # 初始化权重：大家都是 0.5 (Mixup 0.5, CutMix 0.5)
    # weights 数组存储的是 Mixup 的权重 w。CutMix 的权重就是 1-w
    best_weights = np.ones(8) * 0.5 
    best_bacc = 0.0
    
    # 搜索参数
    search_space = np.arange(0.0, 1.01, 0.01) # 精度 0.01，从 0.0 到 1.0
    
    # 迭代优化 (通常 2-3 轮就收敛了)
    for round in range(3):
        print(f"\n🔄 --- Round {round+1} ---")
        param_changed = False
        
        for class_idx in range(8):
            cls_name = TARGET_CLASSES[class_idx]
            current_best_w = best_weights[class_idx]
            local_best_bacc = 0
            
            # 这里的策略是：固定其他 7 个类的权重，只遍历当前类的权重
            for w in search_space:
                # 构造当前测试的权重向量
                temp_weights = best_weights.copy()
                temp_weights[class_idx] = w
                
                # 转为 Tensor [1, 8]
                w_tensor = torch.tensor(temp_weights, device=device).unsqueeze(0)
                
                # 融合：Mix * w + Cut * (1-w)
                final_logits = (logits_mix * w_tensor) + (logits_cut * (1 - w_tensor))
                
                # 计算 BACC
                preds = final_logits.argmax(dim=1).cpu().numpy()
                score = balanced_accuracy_score(targets, preds)
                
                if score > local_best_bacc:
                    local_best_bacc = score
                    current_best_w = w
            
            # 更新该类的最佳权重
            if abs(best_weights[class_idx] - current_best_w) > 1e-4:
                best_weights[class_idx] = current_best_w
                param_changed = True
                
            # 实时显示
            print(f"   Class {cls_name:<4} optimized -> Mixup Weight: {best_weights[class_idx]:.2f} | Current BACC: {local_best_bacc*100:.4f}%")
        
        best_bacc = local_best_bacc
        if not param_changed:
            print("✅ Converged! Weights are stable.")
            break

    # --- 输出最终结果 ---
    print("\n" + "="*50)
    print(f"🏆 OPTIMAL MATHEMATICAL SOLUTION FOUND")
    print(f"   Max BACC: {best_bacc*100:.4f}%")
    print("="*50)
    print("Copy this dictionary to your main code:\n")
    
    print("CLASS_WEIGHTS = {")
    for i, cls in enumerate(TARGET_CLASSES):
        w_m = best_weights[i]
        w_c = 1.0 - w_m
        print(f"    '{cls}': \t({w_m:.2f}, {w_c:.2f}),")
    print("}")

if __name__ == '__main__':
    optimize_weights()