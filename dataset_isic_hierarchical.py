import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit # 🔥 关键引入：按组划分

class ISIC2019_Hierarchical(Dataset):
    def __init__(self, root, train=True, transform=None, test_size=0.2, seed=42):
        """
        ISIC 2019 层级数据集加载器 (MICCAI Standard: Lesion-Level Split)
        """
        self.root = root
        self.transform = transform
        self.train = train
        
        # 1. 路径定义
        csv_path = os.path.join(root, 'ISIC_2019_Training_GroundTruth.csv')
        meta_path = os.path.join(root, 'ISIC_2019_Training_Metadata.csv') # 🔥 必须要有这个
        img_dir = os.path.join(root, 'ISIC_2019_Training_Input')
        
        # 2. 严格的文件检查
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ 缺少标签文件: {csv_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"❌ 缺少元数据文件: {meta_path}\n请去 ISIC 官网下载 'Training Metadata' 并上传到同级目录！")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"❌ 缺少图片文件夹: {img_dir}")
            
        # 3. 读取并合并数据
        df_gt = pd.read_csv(csv_path)
        df_meta = pd.read_csv(meta_path)
        
        # 获取类别名称 (排除 image 列)
        self.classes = df_gt.columns[1:].tolist() 
        
        # 合并表格：为了拿到 lesion_id
        # ISIC 2019 的 image 列名通常是 'image'
        df = pd.merge(df_gt, df_meta[['image', 'lesion_id']], on='image', how='left')
        
        # 填补缺失值：如果某些图没有 lesion_id，就用 image_id 代替，视为独立病灶
        df['lesion_id'] = df['lesion_id'].fillna(df['image'])
        
        # 准备数据数组
        image_ids = df['image'].values
        labels_onehot = df[self.classes].values
        targets_fine = np.argmax(labels_onehot, axis=1)
        groups = df['lesion_id'].values # 🔥 分组依据
        
        # 4. 构建医学语义超类 (Coarse Labels)
        self.coarse_map_dict = {
            'MEL': 0, 'NV': 0,        # Melanocytic
            'BCC': 1, 'AK': 1, 'BKL': 1, 'SCC': 1, # Keratinocytic
            'DF': 2, 'VASC': 2, 'UNK': 2 # Others
        }
        targets_coarse = np.array([self.coarse_map_dict.get(self.classes[i], 2) for i in targets_fine])

        # 5. 🔥🔥🔥 核心修改：按 Lesion 分组划分 (Patient-level Split) 🔥🔥🔥
        # 这样同一个 lesion 的所有图片，要么都在 train，要么都在 test
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        
        # 注意：这里传入 groups 参数
        train_idx, test_idx = next(splitter.split(image_ids, targets_fine, groups))
        
        # 赋值索引
        indices = train_idx if self.train else test_idx
        
        self.image_ids = image_ids[indices]
        self.targets_fine = targets_fine[indices]
        self.targets_coarse = targets_coarse[indices]
        self.img_dir = img_dir
        
        # 6. 打印诊断信息
        if self.train:
            print(f"\n{'='*40}")
            print(f"✅ Dataset Loaded: ISIC 2019")
            print(f"   Split Strategy: [Patient/Lesion-Level Split] (防止数据泄露)")
            print(f"   Mode: {'Train' if train else 'Internal Test'}")
            print(f"   Total Images: {len(self.image_ids)}")
            
            # 检查 VASC 样本数
            if 'VASC' in self.classes:
                v_idx = self.classes.index('VASC')
                v_count = np.sum(self.targets_fine == v_idx)
                print(f"   VASC Samples: {v_count}")
            print(f"{'='*40}\n")
            
    def __getitem__(self, index):
        img_id = self.image_ids[index]
        target_fine = self.targets_fine[index]
        target_coarse = self.targets_coarse[index]
        
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # 容错处理
            img = Image.new('RGB', (320, 320))
            
        if self.transform:
            img = self.transform(img)
            
        return img, target_coarse, target_fine
        
    def __len__(self):
        return len(self.image_ids)