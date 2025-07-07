# Gear Fault Prediction using Machine Learning

This project implements multiple machine learning and deep learning techniques to **predict mechanical gear faults from vibration data**. The goal is to develop an accurate and automated system to classify gear conditions and detect anomalies that may indicate impending failures.

---

## Overview

The workflow includes:

**Data Acquisition & Preprocessing**
- Loading vibration sensor datasets
- Data cleaning (handling missing values, filtering noise)
- Exploratory Data Analysis (EDA) to understand feature distributions and correlations

**Feature Engineering**
- Transforming raw signals into statistical and frequency-domain features

**Modeling**
- **H2O AutoML** to automatically train and tune multiple models (GBM, Random Forest, XGBoost, Deep Learning)
- **Gradient Boosted Machines (GBM)** for interpretable tree-based modeling
- **Convolutional Neural Networks (CNNs)** to capture temporal patterns in vibration signals

**Evaluation**
- Cross-validation for robust performance assessment
- Accuracy metrics (achieving ~80%)
- Confusion matrix analysis to identify misclassification patterns

**Visualization**
- Model performance comparison charts

---

## Repository Contents

- `Copy_of_DA_mini.ipynb`  
  Jupyter/Colab notebook containing:
  - Data cleaning
  - Feature engineering
  - H2O AutoML training
  - GBM and CNN experiments
  - Visualizations and evaluation metrics

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.x
- H2O.ai Python module
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow / keras

*(All dependencies are pre-installed if using Google Colab)*

---


##  Models & Techniques Used

- **H2O AutoML**  
  Automatically trains and tunes a variety of algorithms, selecting the best-performing model.
- **Gradient Boosted Machines (GBM)**  
  Tree-based ensemble method for robust classification.
- **Convolutional Neural Networks (CNN)**  
  For learning temporal patterns directly from signal inputs.

---

## Results

- Achieved ~80% classification accuracy.
- Clear separation between healthy and faulty gear conditions.
- Demonstrated feasibility of combining AutoML and deep learning for vibration-based fault diagnosis.

---

## Future Work

Currently developing an **automated report generation system** to summarize predictions, generate visual analytics, and provide actionable maintenance recommendations.

---

⭐ **If you find this repository helpful, feel free to star it!**

