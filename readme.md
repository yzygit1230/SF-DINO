
<h2 align="center">SF-DINO: Spatial-Frequency Adapted Foundation Model for Multiple Myeloma Diagnosis</h2>


## Overview
<p align='center'>
    <img src="figures/Figure1.png" width="86%" height="86%">
</p>

**Figure 1. Overview of the proposed pipeline. (a) Sample processing and data preparation of multiple myeloma cell images. (b) Flow diagram of the SF-DINO model.**

**_Abstract -_** Microscopic analysis of bone marrow aspirate smears is a critical routine procedure for the initial screening and diagnosis of multiple myeloma (MM). However, manual interpretation is subjective and labor-intensive, further complicated by the fine-grained morphological heterogeneity and inter-class similarities of bone marrow cells. While vision foundation models like DINOv3 have revolutionized natural image understanding, they lack the domain-specific inductive bias required to capture subtle pathological textures in medical cytology. To address these challenges, we first construct a comprehensive, expert-annotated MM diagnosis dataset encompassing six fine-grained cell categories to reflect real-world clinical scenarios. We then propose SF-DINO, a spatial-frequency adapted foundation model that efficiently adapts DINOv3 for MM diagnosis. Specifically, we introduce a parallel spatial-morphology (SM) adapter to inject high-frequency local morphological cues into the backbone without disrupting pre-trained semantics. Additionally, we devise a frequency-selective hashing attention (FSHA) module to model global long-range dependencies efficiently via spectral clustering. Extensive experiments demonstrate that SF-DINO significantly outperforms six state-of-the-art methods, showing impressive generalization and establishing a new benchmark for automated, fine-grained MM diagnosis. Our code and data are available at https://github.com/yzygit1230/SF-DINO.


## Experimental Results

<p align='center'>
    <img src="figures/Figure2.png" width="86%" height="86%">
</p>

**Figure 2. MM diagnosis performance of SF-DINO. (a) t-SNE plot. (b) Confusion matrix. (c) PR curves. (d) ROC curves. (e) Grad-CAM visualization. Best viewed in color.**

### Dataset Preparation
Put the dataset as follows:
```text
DatasetMM
├── 0
│   ├── 1.png
│   ├── 2.png
│   ├── ...
├── 1
│   ├── 1.png
│   ├── 2.png
│   ├── ...
...
```

### Five-fold Cross-validation

Train five stratified folds. Each fold uses part of its training portion as a
validation set for selecting the best checkpoint, and uses the held-out fold
only for the final test:

```bash
python train.py --dataset_root /path/to/DatasetMM --epochs 300 --batch_size 24
```

Re-evaluate all saved best checkpoints and generate out-of-fold metrics:

```bash
python test.py --dataset_root /path/to/DatasetMM --checkpoint_dir outputs/5fold
```

### Dataset Access
Due to patient privacy protections, access to the **MM diagnosis dataset** can be requested by contacting the corresponding author with a reasonable research justification. For any questions or collaborations, please contact [Liye Mei](mailto:liyemei@whu.edu.cn), [Cheng Lei](mailto:leicheng@whu.edu.cn).
