````markdown
# **LFMT: Light Field Image Super-Resolution**

Official PyTorch implementation of  
**"Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution"**

This work proposes **LFMT**, a hybrid Mamba-Transformer framework for light field image super-resolution (LFSR). LFMT explores non-local spatial-angular correlations and disparity-aware epipolar structures through the proposed **Subspace Simple Scanning (Sub-SS)** strategy and a dual-stage spatial-angular/EPI-domain modeling framework.

---

## 📰 News

✅ **Accepted by Pattern Recognition**:  
Our paper **"Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution"** has been accepted by **Pattern Recognition**.

✅ This repository contains the official PyTorch implementation of **LFMT**.

<!-- 
![Results](figs/results.png)
-->

---

## 🔍 Overview

LFMT is designed to efficiently model the complex 4D correlations in light fields. The main components include:

- **Subspace Simple Scanning (Sub-SS)**: reduces redundant directional propagation within highly correlated LF subspaces.
- **SA-RSMB**: performs spatial-angular dependency aggregation in spatial and angular domains.
- **EPMB + EPTB**: jointly refine disparity-aware epipolar structures using Mamba-based sequential propagation and Transformer-based global interaction.
- **Hybrid Mamba-Transformer Framework**: achieves a favorable balance between reconstruction performance and computational efficiency.

---

## **Requirements**

To set up the environment, install the following dependencies:

- **Python**: `3.9`
- **PyTorch**: `2.1.1`
- **Torchvision**: `0.16.1`
- **CUDA**: `11.8`
- **Additional Packages**:
  - `causal-conv1d==1.1.1`
  - `mamba-ssm==1.0.1`
  - `numpy`
  - `scipy`
  - `h5py`
  - `opencv-python`
  - `einops`
  - `tqdm`

### **Installation**

```bash
conda create -n lfmt python=3.9
conda activate lfmt

pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install causal-conv1d==1.1.1 mamba-ssm==1.0.1
pip install numpy scipy h5py opencv-python einops tqdm
````

---

## **Datasets**

We use five commonly adopted LFSR benchmark datasets following previous works:

* **EPFL**
* **HCInew**
* **HCIold**
* **INRIA**
* **STFgantry**

The datasets can be prepared following [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR).

Please place the downloaded datasets in:

```bash
./datasets/
```

---

## **Training**

### **1. Prepare Training Data**

Generate the training data by running:

```bash
python Generate_Data_for_Training_aug.py
```

The processed training data will be saved in:

```bash
./data_for_training/
```

### **2. Start Training**

To train LFMT, run:

```bash
python train.py
```

Model checkpoints and training logs will be saved in:

```bash
./log/
```

---

## **Testing**

### **1. Prepare Test Data**

Generate the test data by running:

```bash
python Generate_Data_for_Test.py
```

The processed test data will be saved in:

```bash
./data_for_test/
```

### **2. Start Testing**

Run the following command to test LFMT on each benchmark dataset:

```bash
python test.py
```

The output `.mat` files will be stored in:

```bash
./Results/
```

### **3. Generate SR Images**

To generate super-resolved RGB images, run:

```bash
python GenerateResultImages.py
```

The generated images will be saved in:

```bash
./SRimage/
```

### **4. Generate Combined Visualization Images**

To create combined comparison images, run:

```bash
python img_combine.py
```

The results will be saved in:

```bash
./combined_Test_Real/
./combined_Test_Synth/
```

---

## **Pretrained Models**

The pretrained LFMT model weights will be released here:

🔗 [Click here to download](YOUR_PRETRAINED_MODEL_LINK)

Please place the downloaded model weights in:

```bash
./pretrained/
```

---

## **Results**

LFMT achieves state-of-the-art reconstruction performance on widely used LFSR benchmark datasets while maintaining favorable computational efficiency.

<!-- You may add quantitative results here, for example:

| Method | Scale | Params. | FLOPs | Ave. PSNR/SSIM |
|--------|-------|---------|-------|----------------|
| LFMT   | x2    | 2.04M   | 62.76G | 39.63/0.9881 |
| LFMT   | x4    | 2.19M   | 66.72G | 32.66/0.9471 |

-->

---

## **Notice**

* Please modify the file loading and saving paths according to your local environment.
* The dataset preparation follows the commonly used LFSR benchmark protocol.
* For different scale factors, please check and modify the corresponding configuration in the training and testing scripts.
* The pretrained models and additional results will be updated soon.

---

## 📖 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{Liu_2026_PR_LFMT,
  title={Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution},
  author={Liu, Haosong and Zhu, Xiancheng and Zeng, Huanqiang and Zhu, Jianqing and Cao, Jiuwen and Hou, Junhui},
  journal={Pattern Recognition},
  pages={114210},
  year={2026},
  publisher={Elsevier}
}
```
---
## 📬 Contact

If you have any questions, please feel free to open an issue or contact us:

* 📧 [hsliu@stu.hqu.edu.cn](mailto:hsliu@stu.hqu.edu.cn)

```
```

