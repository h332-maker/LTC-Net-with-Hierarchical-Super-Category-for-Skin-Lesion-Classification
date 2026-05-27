import torch
import torch.nn as nn
import timm

class ResNet_Hierarchical(nn.Module):
    def __init__(self, num_fine_classes=9, num_coarse_classes=3, use_cosine=True, model_name='resnet50'):
        super(ResNet_Hierarchical, self).__init__()
        
        print(f"🏗️ [Model] Init {model_name} (512x512) with Dual Heads...")
        
        # 1. 加载 ResNet50
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0, 
            global_pool='' 
        )
        
        # 自动获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy)
            self.in_features = features.shape[1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.use_cosine = use_cosine
        
        # 2. 双头分类器
        if use_cosine:
            self.linear_fine = nn.Linear(self.in_features, num_fine_classes, bias=False)
            self.linear_coarse = nn.Linear(self.in_features, num_coarse_classes, bias=False)
            self.s = 30.0
        else:
            self.linear_fine = nn.Linear(self.in_features, num_fine_classes)
            self.linear_coarse = nn.Linear(self.in_features, num_coarse_classes)

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        if self.use_cosine:
            x_norm = torch.norm(features, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            
            # Fine Head
            w_norm_fine = torch.norm(self.linear_fine.weight, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            features_n = features / x_norm
            weight_n_fine = self.linear_fine.weight / w_norm_fine
            logits_fine = torch.mm(features_n, weight_n_fine.t()) * self.s
            
            # Coarse Head
            w_norm_coarse = torch.norm(self.linear_coarse.weight, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            weight_n_coarse = self.linear_coarse.weight / w_norm_coarse
            logits_coarse = torch.mm(features_n, weight_n_coarse.t()) * self.s
        else:
            logits_fine = self.linear_fine(features)
            logits_coarse = self.linear_coarse(features)
            
        return logits_fine, logits_coarse