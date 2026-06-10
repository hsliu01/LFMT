## 📰 News

✅ **Jun 10, 2026**: Our work **"Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution"** has been **accepted by Pattern Recognition**. 

## **Requirements**
To set up the environment, install the following dependencies:

- **PyTorch**: `2.1.1`
- **Torchvision**: `0.16.1`
- **Python**: `3.9.19`
- **CUDA**: `11.8`
- **Additional Packages**:
  - `causal-conv1d==1.1.1`
  - `mamba-ssm==1.0.1`

### **Installation**
You can create the environment using:
```bash
pip install torch==2.1.1 torchvision==0.16.1
pip install causal-conv1d==1.1.1 mamba-ssm==1.0.1
```

---

## **Training**
### **1. Prepare Training Data**
We utilize five Light Field  benchmarks from [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR):
- **EPFL**
- **HCInew**
- **HCIold**
- **INRIA**
- **STFgantry**

Download the datasets and place them in the `./datasets/` directory.

To generate the training data, run:
```bash
python Generate_Data_for_Training.py
```
The processed training data will be saved in `./data_for_training/`.

### **2. Start Training**
To train the network, run:
```bash
python train.py
```
Model checkpoints will be saved in `./log/`.

---

## **Testing**
### **1. Prepare Test Data**
Generate the test data by running:
```bash
python Generate_Data_for_Test.py
```
The processed test data will be saved in `./data_for_test/`.

### **2. Start Testing**
Perform testing on each dataset using:
```bash
python test.py
```
The output `.mat` files will be stored in `./Results/`.

To generate SR RGB images, run:
```bash
python GenerateResultImages.py
```
The generated images will be saved in `./SRimage/`.

---

## **Notice**
- The pretrained model weights can be downloaded from the link below:

🔗 [Click here to download](https://drive.google.com/drive/folders/1e6egLNUk2qidbdlIJ4YzX1z92TjiY1G6?usp=drive_link).

## 📖 Citation

If you find this work helpful, please consider citing the following paper:

```bibtex
@article{Liu_2026_PR_LFMT,
title={Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution},
author={Liu, Haosong and Zhu, Xiancheng and Zeng, Huanqiang and Zhu, Jianqing and Cao, Jiuwen and Hou, Junhui},
journal={Pattern Recognition},
year={2026},
publisher={Elsevier} }
```

If you have any questions, please pull an Issue and feel free to contact me at:

- 📧 hsliu@stu.hqu.edu.cn
