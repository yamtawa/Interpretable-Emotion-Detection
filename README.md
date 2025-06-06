# **NLP_PROJECT** 🚀

Welcome to our **final NLP project** – a **sensational, groundbreaking, and amazing** effort in emotion classification using **BERT**! This repository contains everything you need to **train, evaluate, and fine-tune** our model.

## **🚀 How to Run the Main Pipeline**
Follow these steps to train and evaluate the model or explore neurons activations:

### **Step 1: Configure the Pipeline**
Locate the configuration file:
```bash
/configs/config.yaml
```
This file contains all **hyperparameters** and **training settings**.

### **🔧 Configuration Details**
#### **General Settings**
- **Saved Model Name**: `"bert-base-uncased_try1"`
- **Dataset Options**: Choose from:
  - `"dair-ai/emotion"`
  - `"jeffnyman/emotions"`
  - `"go_emotions"`
- **Batch Size**: `16`
- **Base Model**: `"bert-base-uncased"`

#### **PIPLINE**
This section defines multiple **steps** of a pipline, where each step includes:
- **TITLE**: The title of the step that is about to be done (works now for: "MODEL_TRAINING" or "NEURON_EXPLORATION" )
- **ACTIVATE**: States weather to preform this step in the current pipline
#### **Neuron Exploration Plan**

- **Criterion Name**: Loss function (e.g., `CE` for CrossEntropy).
  - **Scheduler**: `CosineAnnealingLR` (currently default).
  - **Optimizer**: `Adam`.
  - **Learning Rates**:
    - `LR_FEATURES`: `0.0001`
    - `LR_HEAD`: `0.0001`
  - **Regularization**:
    - `WEIGHT_DECAY`: `0.01`
    - `MOMENTUM`: `0.1`
  - **Training Parameters**:
    - `NUM_EPOCHS`: `20`
    - `TRAINING_PHASE`: Choose from:
      - `"train_from_scratch"`
      - `"retrain"`
      - `"eval"`
    - **Fine-Tuning Mode**:
      - `True`: Fine-tune only the heads.
      - `False`: Train the entire model.

#### **Neuron Exploration Plan**
- **Step 2 (NEURON_EXPLORATION)**:
  - **WANTED_LABELS**: Choose specific labels (e.g., `['anger','fear']`) or `"all"`.
  - **NEURONS_FUNCTION_NAME**: Function to analyze neuron activations.
  - **CRITERION_NAME**: `CE`
  - **OPTIMIZER_NAME**: `Adam`
  - **Fine-Tune Mode**: `False`

---

### **Run the pipline**
After configuring the wanted steps, run the following command with the :
```bash
python main.py
```
This will preform the wanted pipline

---

### **📌 Additional Notes**
- Ensure you have all required dependencies installed before running:
  ```bash
  pip install -r requirements.txt
  ```
- The model weights are automatically saved under:
  ```bash
  models_weights/bert-base-uncased_try1.pth
  ```

---

### **📢 Authors & Contributors**
If you have any questions, don't ask me! Understand by yourself.

🚀 Happy training! 🚀

---
