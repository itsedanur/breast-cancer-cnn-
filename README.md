#  Breast Cancer Classification with CNN & Transfer Learning

This project focuses on **classifying breast cancer images as benign or malignant** using Convolutional Neural Networks (CNN) and Transfer Learning techniques with Keras and TensorFlow.

## Dataset

The dataset is structured as:

data/
├── train/
│ ├── benign/
│ └── malignant/
├── val/
│ ├── benign/
│ └── malignant/
└── test/
├── benign/
└── malignant/



The original dataset is split into train/val/test using a custom script (`split_dataset.py`).

---

## 🔧 Project Files

| File | Description |
|------|-------------|
| `prepare_dataset.py` | Prepares and checks the image dataset |
| `split_dataset.py`   | Splits dataset into train/val/test |
| `train_cnn.py`       | Trains a simple CNN model |
| `train_transfer.py`  | Transfer learning with MobileNetV2 |
| `fine_tune_transfer.py` | Fine-tunes the pretrained model |
| `evaluate_model.py`  | Evaluates the basic CNN |
| `evaluate_transfer.py` | Evaluates the transfer learning model |
| `evaluate_finetune.py` | Evaluates the fine-tuned model |
| `breast_cancer_model.h5` | CNN trained model |
| `transfer_model.h5` | Transfer learning model |
| `fine_tuned_model.h5` | Fine-tuned model |

---

# Model Performances

Confusion matrices are plotted for each model:
-  Basic CNN: Accuracy ~ 69%
- Transfer Learning: Accuracy ~ 42%
-  Fine-Tuning: Accuracy ~ 51%

You can find them in the evaluation scripts.

---

## Requirements

```bash
pip install tensorflow matplotlib scikit-learn


