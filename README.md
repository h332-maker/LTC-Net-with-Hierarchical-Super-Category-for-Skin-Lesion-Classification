# LTC-Net: Long-Tailed Calibration Network with Hierarchical Super-Category for Skin Lesion Classification

Official implementation of the paper **"Long-Tailed Calibration Network with Hierarchical Super-Category for Skin Lesion Classification"**.

---

## 🌟 Overview

The automated diagnosis of skin lesions is severely hindered by extreme long-tailed class imbalances. Current deep learning paradigms predominantly treat different pathological dermatoses as flat, isolated, and mutually exclusive labels, neglecting their intrinsic biological taxonomy and hierarchical relationships. Such structural oversight deprives data-starved rare tail classes of essential semantic guidance, leading to feature collapse and biased decision boundaries.

**LTC-Net** is a systematic, taxonomy-driven framework that progressively couples feature robustness with decision fairness. By leveraging a clinical super-category hierarchy based on histopathological lineage, LTC-Net bridges algorithmic regularizations with medical domain knowledge across the feature, manifold, and decision spaces.

### 💡 Key Highlights
- **Semantic Anchoring Topology (SAT):** Embeds histopathological homology within a dual-head architecture (coarse super-category head + fine disease head) to anchor sparse tail lesions to stable super-class manifolds, fundamentally preventing feature collapse.
- **Multi-dimensional Manifold Alignment (MMA):** Intervenes in the latent space by synergizing global Mixup and local CutMix via a super-category conditioned routing mechanism to sharpen blurred decision boundaries without overfitting.
- **Distribution-Perception Probability Calibration (DPC):** Functions as a highly efficient, training-free analytical post-hoc operator to shift decision boundaries via frequency-aware margins, eliminating systematic prediction bias.
- **Superior Clinical Reliability:** Comprehensively breaks the traditional "accuracy paradox" of long-tailed learning, simultaneously maximizing equitable minority class retrieval (high-risk, ultra-rare malignancies) and overall precision.

---

## 🗺️ Framework Architecture

```
  Input Image ---> [ MMA Module ] ---> [ Shared Encoder ] 
                         |                     |
                (Global/Local Routing)         v
                         |             +-------+-----------------------------+
                         v             | SAT Module                          |
                Manifold Completion    |   ├── Coarse Head -> Super-Category  |
                         |             |   └── Fine Head   -> Disease Class  |
                         v             +------------------------+------------+
                  Dense Features                                |
                                                                v
                                                       Uncalibrated Logits
                                                                |
                                                                v
                                                       [ DPC Module ] (Training-free)
                                                                |
                                                                v
                                                       Equitable Diagnostic Scores
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Linux / Windows
- Python >= 3.8
- PyTorch >= 1.12 (CUDA supported)
- `torchvision`, `scikit-learn`, `tqdm`, `numpy`

### 2. Installation
Clone this repository and install dependencies:
```bash
git clone [https://github.com/yourusername/LTC-Net.git](https://github.com/yourusername/LTC-Net.git)
cd LTC-Net
pip install -r requirements.txt
```

### 3. Data Preparation
Our manual pathology-driven taxonomy splits the dataset into 4 clinical biological super-categories based on cellular origin and malignancy:
1. **Melanocytic ($c_{mc}$):** Melanoma (MEL), Melanocytic nevus (NV)
2. **Keratinocytic ($c_{kc}$):** Actinic keratosis (AKIEC), Benign keratosis (BKL)
3. **Cell Carcinoma ($c_{cc}$):** Basal cell carcinoma (BCC), Squamous cell carcinoma (SCC)
4. **Mesenchymal/Vascular ($c_{mv}$):** Dermatofibroma (DF), Vascular lesions (VASC)

Organize your data directories following the structure below:
```text
├── data/
│   ├── ISIC2018/
│   └── ISIC2019/
│       ├── train/
│       ├── val/
│       └── test/
```

### 4. Training Pipeline
Run the joint optimization with Semantic Anchoring Topology (SAT) and Multi-dimensional Manifold Alignment (MMA):
```bash
python train.py \
    --dataset ISIC2019 \
    --data_dir ./data/ISIC2019 \
    --model resnet50 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --lambda_sat 0.62 \
    --output_dir ./checkpoints/ltc_net_base/
```

### 5. Post-hoc Bivariate Grid Calibration (DPC)
Execute the training-free analytical calibration phase on the validation set priors to find optimal boundaries:
```bash
python calibrate.py \
    --checkpoint ./checkpoints/ltc_net_base/best_model.pth \
    --dataset ISIC2019 \
    --tau_bounds 0.1 1.0 0.05 \
    --gamma_bounds 0.5 2.0 0.1
```

---

## 📊 Experimental Results

### 1. Main Benchmarks on Clinical Skin Lesions (%)

| Method | ISIC2018 (BACC) | ISIC2018 (OA) | ISIC2018 (F1) | ISIC2019 (BACC) | ISIC2019 (OA) | ISIC2019 (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CE Baseline | 68.83 | 84.68 | 69.55 | 77.30 | 86.37 | 78.39 |
| Focal Loss | 70.41 | 84.34 | 71.10 | 78.87 | 87.40 | 80.94 |
| G-PACO | 75.44 | 77.77 | 62.71 | 86.03 | 86.07 | 76.81 |
| ProCo | 76.40 | 76.53 | 65.29 | 83.75 | 86.72 | 79.39 |
| DSCL | 76.36 | 84.30 | 70.75 | 85.84 | 89.96 | 85.85 |
| LOS | 76.05 | 84.78 | 72.95 | 86.22 | 83.56 | 72.46 |
| DeiT-LT | 77.34 | 86.36 | 74.30 | 86.26 | 90.70 | 86.41 |
| CBTS | 75.03 | 83.94 | 71.11 | 86.27 | 90.64 | 86.36 |
| **LTC-Net (Ours)** | **78.79** | **87.01** | **75.09** | **90.23** | **90.95** | **86.55** |

### 2. External Generalization on Natural Images (Overall Accuracy %)

| Method | CIFAR100-LT (IF=100) | CIFAR100-LT (IF=50) | CIFAR100-LT (IF=10) | CIFAR10-LT (IF=100) | CIFAR10-LT (IF=50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CE Baseline | 38.3 | 43.9 | 55.7 | 70.4 | 74.8 |
| LDAM-DRW | 42.0 | 46.6 | 58.7 | 77.0 | 81.0 |
| BBN | 42.6 | 47.0 | 59.1 | 79.8 | 82.2 |
| CSA | 46.6 | 51.9 | 62.6 | 82.5 | 86.0 |
| **LTC-Net (Ours)** | **50.8** | **54.8** | **63.5** | **84.4** | **86.7** |

### 3. Core Ablation Analysis (ISIC2019)

| Configuration | SAT | MMA | DPC | BACC (%) | OA (%) | F1-Score (%) | AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | | | | 77.29 | 86.37 | 78.40 | 97.70 |
| + SAT | \checkmark | | | 79.50 | 87.91 | 81.56 | 98.10 |
| + SAT + MMA | \checkmark | \checkmark | | 86.61 | 90.58 | 86.24 | 98.14 |
| **LTC-Net (Full)** | \checkmark | \checkmark | \checkmark | **90.23** | **90.95** | **86.55** | **98.21** |

### 4. Evaluation of Taxonomy Grouping Strategies

| Strategy | Accuracy | Macro-F1 | Macro-Prec | Macro-Recall |
| :--- | :---: | :---: | :---: | :---: |
| Random Grouping | 0.8690 | 0.8007 | 0.8332 | 0.7706 |
| Visual Similarity | 0.8714 | 0.8192 | 0.8441 | 0.7958 |
| Sample Balanced | **0.8746** | 0.8203 | 0.8470 | 0.7953 |
| **Ours (Pathology)** | 0.8733 | **0.8322** | **0.8614** | **0.8048** |

---

## 📂 Repository Structure
```text
├── models/
│   ├── resnet.py            # Feature extractor backbone
│   ├── sat_head.py          # Dual-head semantic anchoring modules
│   └── mma_routing.py       # Multi-dimensional manifold alignment mechanics
├── utils/
│   ├── taxonomy.py          # Histopathological lineage definition rules
│   ├── calibration.py       # DPC distribution probability operations
│   └── augmentations.py     # Global Mixup and local CutMix helpers
├── train.py                 # Core training loop execution script
├── calibrate.py             # Validation-set bivariate grid validation runner
├── requirements.txt         # Package dependencies file
└── README.md                # Project documentation
```

---

## ✉️ Citation

If you find LTC-Net helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{ltcnet2026,
  title={Long-Tailed Calibration Network with Hierarchical Super-Category for Skin Lesion Classification},
  author={Anonymous Author(s)},
  booktitle={Proceedings of Lecture Notes in Computer Science (LNCS)},
  year={2026}
}
```
