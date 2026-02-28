# Music Genre Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)    [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)  [![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)  [![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/)
This project aims to classify audio into multiple genres. Ten possible genres are:
 **blues, classical, country, disco, hiphop, jazz, metal, pop, reggae** and **rock**
 
## Key Libraries / Modules
* PyTorch
* Torchaudio
* Librosa
* Transformers
* Wandb

## Repository Structure

```bash

├── DL-23f3003877-notebook-t12026.ipynb # Final kaggle submission notebook
├── LICENSE                             # Open-source license
├── README.md                           # Project documentation
├── notebooks/                          # EDA and training notebooks
│   ├── EDA.ipynb                       # Exploratory Data Analysis
│   ├── Data_Generator/                 # To run data generator script in kaggle
│   │   ├── bpm-calculator.ipynb        # Pre-compute bpm for faster data creation
│   │   └── mel-spectrogram_generators/
│   │       ├── Data_generator-1.ipynb
│   │       ├── Data_generator2.ipynb
│   │       ├── Data_generator3.ipynb
│   │       ├── Data_generator4.ipynb
│   │       └── Data_generator5.ipynb
│   └── models/                         # Prototyping model architectures and training models
│       ├── AST_model.ipynb
│       ├── ResNet_50.ipynb
│       ├── scratch_cnn.ipynb
│       └── Scratch_vision_transformer.ipynb
└── src/                                # Modular, production-ready source code
    ├── Data_generator/                 # Utility scripts for audio processing
    │   ├── bpm_calculator.py
    │   └── data_generator.py
    ├── dataset/                        # Dataset classes and dataloaders
    │   ├── test_dataset.py
    │   ├── test_dataset_for_AST.py
    │   ├── train_val_dataset.py
    │   └── train_val_dataset_for_AST.py
    ├── inference/                      # Scripts for prediction on test data
    │   ├── ensemble_inference.py
    │   └── inference.py
    ├── models/                         # Model class definitions
    │   ├── AST.py
    │   ├── Resnet_50.py
    │   ├── scratch_CNN.py
    │   └── scratch_vision_transformer.py
    └── Trainers/                       # Custom training loops and optimization logic
        └── custom_trainer.py
```
## Workflow
```
Data -> Data Preprocessing and Dataset creation -> Model configuration -> Training and Validation -> Save best checkpoint -> Predictions.
```

## 📊 Project Overview

### Problem Statement

**Predict the correct genre label** for each noisy mashup. The model should be capable of handling  Instrument balance changes,   Cross-song stem recombination, Tempo variations, added noise and songs of different durations.

### Key Results
Model | Parameters | Macro F1 | Best For | Inference time (3k samples)
|--    |-----------|----------|-----------------|--------------|
**Scratch CNN**| 6.3M | 0.98540 | Fast, light and decent accuracy | 1min
|**ResNet-50**| 23.5M| 0.98729  | Fast, moderate and Great accuracy | 1min 6sec
|**AST**| 86.2M| 0.98720 | slow, heavy and Great accuracy | 6min 30 sec

**Scratch CNN** and  **ResNet-50** ensemble scored 0.99068 while taking 2min for 3k samples.

**Scratch CNN** , **ResNet-50** and **AST** ensemble scored 0.99599 while taking 7min 30sec for 3k samples.

**Kaggle Competition Performance:**

-   Private Leaderboard:  **NOT RELEASED YET** , ** **
-   Public Leaderboard:  **0.99599** , **RANK 1**

# **Architecture Details**

## 1. Scratch CNN
Simple CNN with  6.3M parameters it is faster to train and inference while having performance extremely close or better then other models with much more parameters.
<img width="3232" height="1312" alt="Gemini_Generated_Image_eh3y5eh3y5eh3y5e" src="https://github.com/user-attachments/assets/0538eddf-5681-426a-9ecb-87baf41b736e" />


### **Layer details:**
 * **2D conv** (find pattern in images)
 * **Batch normalization** (prevent vanishing gradient and helps to converge faster)
 *  **Relu** (add non-linearity)
 * **2D max pool** (Downsamples spatial dimensions to reduce computational complexity)
 * **Dropout** (Applies regularization to prevent overfitting on the training data.)

### **Classification head:**
* Adaptive Average Pool (calculate avg to find 1 float for each channel)
* Flatten (Flatten all 512 floats)
* Linear Classifier (512 -> 10 -> genre logits)   

### **Training Strategy:**
*   Batch size 124.
*   Learning rate of 0.001 with weight decay of 0.01 and reduce lr with a factor of 0.5.
*   Training data samples 101771.
*   Validation data samples 25452.

  

## 2. **ResNet-50**

Residual network having 23.5M parameters with skipped connections to make deep architecture possible. 

<img width="1351" height="722" alt="3a93c9fd" src="https://github.com/user-attachments/assets/fb28c2ef-de49-4345-86e9-743e40e7c6db" />


### **Architecture Details:**
* The model consists of **50 layers**, primarily organized into convolutional "Bottleneck" blocks with skip connections.
* Skip (or residual) connections prevent the vanishing gradient problem, which helps in effectively training much deeper models.
* **Steps inside each Bottleneck block:**
  1. **1x1 Conv layer:** Reduces input channels for more efficient and faster calculations.
  2. **Batch Norm & ReLU:** Normalizes data and adds non-linearity.
  3. **3x3 Conv layer:** Extracts spatial features and finds patterns in the images.
  4. **Batch Norm & ReLU:** Normalizes and adds non-linearity.
  5. **1x1 Conv layer:** Increases input channels to match the required dimensions for the next block.
  6. **Skip Connection:** The original block input is added directly to this output before the final ReLU activation.

### **Training Strategy:**
* **Batch Size:** 64
* **Learning Rate:** Initial rate of `0.001` with a weight decay of `0.01`. A learning rate scheduler is used to reduce the LR by a factor of `0.5`.
* **Training Data:** 101,771 samples
* **Validation Data:** 25,452 samples


## 3. **AST (Audio Spectrogram Transformer)**
Audio Spectrogram Transformer with 86.2M parameters. A convolution free fully attention based model trained on audio log mel-spectrograms.

<img width="401" height="331" alt="image" src="https://github.com/user-attachments/assets/6231a2af-3d93-4d0f-a2ad-3ce3e9eed8c0" />

### **Model Architecture:**
* Audio log mel-spectrograms with 128 frequency bands is used as an input.
* Spectrogram is split into 16x16 patches with an overlap of 6 in both time and frequency dimension.
* Each 16x16 patch is flatten to make a embedding of size 768.
* A learnable positional embedding of size 768 is added to each embedding
* A [CLS] token is appended at the beginning of the sequence. This [CLS] token will be used in classification layer at the end of the transformer sequences.
* Sequence is passed to 12 transformer encoding layers.
* [CLS] token is retrieved from the sequence and then we apply a classification layer to get logits for all 10 classes.

### **Training Strategy:**
* Batch size 32.
* Learning rate of 5e-5 for head/classifier and 1e-5 for backbone/transformer layers.
*   Training data samples 101771.
*   Validation data samples 25452.


