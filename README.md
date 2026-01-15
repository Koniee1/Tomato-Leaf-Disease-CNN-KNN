# Tomato-Leaf-Disease-CNN-KNN
## Hybrid CNN–KNN Model for Tomato Leaf Disease Classification
Implementation of CNN-KNN hybrid model


This repository contains the implementation of a hybrid deep learning
framework combining EfficientNetV2B1 for feature extraction and
K-Nearest Neighbors (KNN) for disease classification.

### Dataset
The dataset used in this study was obtained from the Mendeley Data Repository.
Due to licensing constraints, the dataset is not included in this repository.

### Method Overview
- Pretrained EfficientNetV2B1 for feature extraction
- SMOTE for class imbalance handling
- PCA for dimensionality reduction (95% variance retained)
- KNN with GridSearchCV for classification
- Grad-CAM++ for model interpretability

### Requirements
See `requirements.txt`
