# Deepfake Detection with Frequency-Enhanced Self-Blended Images

This research project is an extension of the following project: [Frequency-Enhanced Self-Blended Images](https://github.com/gufranSabri/FSBI/tree/main). We explore alternative color spaces to see if they help to improve the deepfake detection process.

# Datasets

* Access the FaceForensics++ dataset by completing this [form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform).
* Access the Celeb-DF-v2 dataset by completing this [form](https://docs.google.com/forms/d/e/1FAIpQLScoXint8ndZXyJi2Rcy4MvDHkkZLyBFKN43lTeyiG88wrG0rA/viewform).

# Project Instructions

This project is compatible with a limited number of Python versions.

**Python 3.8.0** has been tested and is recommended for use.

To be able to install all the requirements for the project, be sure that your **pip, setuptools, and wheel** are up-to-date.

### Model training

To train the model on Faceforencics++ dataset, use  `init_ff()` in `ESBI_Dataset()` class (located at src\utils\esbi.py, line 36).

To train the model on Celeb-DF-v2 dataset, use `init_cdf()` in `ESBI_Dataset()` class (located at src\utils\esbi.py, line 36).

Train with the following command:

`python src/train_sbi.py src/configs/sbi/base.json -n exp -w sym2 -m reflect -e 100`

This will start the training with a session name "exp", "sym2" wavelet, "reflect" mode and 100 epochs.

### Model Testing

To run inference on the FaceForensics++ dataset using a trained model, use the following command:

`python src/inference/inference_dataset.py -d FF -t Face2Face`

This command tests the model on Face2Face-manipulated videos. To test on other manipulation methods (Deepfakes, FaceSwap, NeuralTextures), replace Face2Face with the desired method name. Make sure the model weights you want to test with are placed in a folder named "weights".

To run inference on the Celeb-DF-v2 dataset using a trained model, use the following command:

`python inference_dataset.py -d CDF`

### Color Space Change

To switch between color spaces uncomment the code chunks that have the corresponding color space name above them in the following files:

* src\utils\esbi.py
* src\inference\inference_dataset.py
* src\inference\generate_misclassifications.py
* src\generate_cam.py

---

There are two README files in the original FSBI project. Both of them are presented below.

# README 1

---

READ PAPER `<a href="https://www.sciencedirect.com/science/article/pii/S026288562500006X?via%3Dihub">`[HERE]`</a>`

# Deepfake detection using Enhanced Self Blended Images using DWT features

Welcome! This research project is an extension of the following project: [Self Blended Images](https://github.com/mapooon/SelfBlendedImages/tree/master)
 `<br>`

Our best model weights can be downloaded at [this link](https://www.kaggle.com/datasets/gufransabri3/fsb-best-model).

### Approach

![Image Description](fig/diagrams/ESBI.png) `<br>`

This project aims to detect deepfakes using Self-Blended image (SBI) approach. The process for creating the self blended images is similar to the reference paper ([Self Blended Images](https://github.com/mapooon/SelfBlendedImages/tree/master)).

We enhance the SBI method using Discrete Wavelet Transform features. We apply the sym2 transform with reflect mode. DWT features emphasize the frequency features in the deepfakes. Frequency artifacts are common in deepfakes.
The approach is illustrated in the diagram above.

### Results

| Method                        | AUC (%)         |
| ----------------------------- | --------------- |
| MCX-API                       | 90.87           |
| FRDM                          | 79.40           |
| PCL + I2G                     | 90.03           |
| LRL                           | 78.26           |
| LipForensics                  | 82.40           |
| FTCN                          | 86.90           |
| EFNB4 + SBI (FF RAW)          | 93.18           |
| EFNB4 + SBI (FF c23)          | 92.93           |
| **EFNB5 + ESBI [ours]** | **95.49** |

---

# README 2

---

# Face-X-ray

The author's unofficial PyTorch re-implementation of Face Xray

This repo contains code for the BI data generation pipeline from  [Face X-ray for More General Face Forgery Detection](https://arxiv.org/abs/1912.13458) by Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, Baining Guo.

# Usage

Just run bi_online_generation.py and you can get the following result. which is describe at Figure.5 in the paper.

![1746631812262](image/README/1746631812262.jpg)

To get the whole BI dataset, you will need crop all the face and compute the landmarks as describe in the code.
