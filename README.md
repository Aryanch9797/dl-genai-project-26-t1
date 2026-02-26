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
Model | Parameters | Macro F1 | Inference Speed | Best For
|--    |-----------|----------|---------------|-----------------|
**Scratch CNN**| 6.3M | 0.98540 | Fastest |  light and decent accuracy
|**ResNet-50**| 23.5M| 0.98729  | Fast  |  moderate and Great accuracy
|**AST**| 86.2M| 0.98322| moderate | heavy and underperforming 

**Scratch CNN** and  **ResNet-50** ensemble scored 0.99068 while taking  2x less time for inference then **AST**

**Scratch CNN** , **ResNet-50** and **AST** ensemble scored 0.99070 while taking time  3x more then **Scratch CNN** and  **ResNet-50** ensemble .

**Kaggle Competition Performance:**

-   Private Leaderboard:  **NOT RELEASED YET** , ** **
-   Public Leaderboard:  **0.99070** , **RANK 1**

**Architecture Details**

1. Scratch CNN
Simple CNN with  6.3M parameters it is faster to train and inference while having performance extremely close or better then other models with much more parameters.
