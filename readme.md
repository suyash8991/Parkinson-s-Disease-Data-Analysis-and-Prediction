# Parkinson's Disease Prediction Project

## Project Overview

This project focuses on developing machine learning models to classify Parkinson's Disease using voice characteristics. By analyzing vocal biomarkers, the aim is to create an accurate predictive tool for early Parkinson's Disease detection.

## Dataset

### Source
The dataset is sourced from Kaggle, originally published in the paper:
- Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM
- "Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection"
- BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

### Data Characteristics
- Total Samples: 195 rows
- Features: 24 columns
- Key Features:
  - Vocal fundamental frequency measurements
  - Jitter and shimmer variations
  - Noise-to-harmonic ratio
  - Nonlinear dynamical complexity measures

## Methodology

### Preprocessing
- Data cleaning and preparation
- Feature scaling
- Handling class imbalance

## Model Performance

The project generates a comprehensive `model_comparison_report.txt` in the `results/` directory, which includes:
- Classification metrics
- Confusion matrix details
- Best hyperparameters for each model
- ROC AUC scores

### Model Comparison
We evaluated three machine learning models:
| Model           | Precision | Recall  | F1-Score | ROC AUC | Best Hyperparameters              |
|---------------|-----------|--------|----------|--------|----------------------------------|
| Random Forest | 0.8974    | 0.8974 | 0.8974   | 0.9345 | n_estimators: 299, max_depth: 12 |
| Decision Tree | 0.8770    | 0.8718 | 0.8737   | 0.8690 | max_depth: 14, min_samples_split: 8 |
| XGBoost       | 0.9219    | 0.9231 | 0.9217   | 0.9103 | n_estimators: 132, max_depth: 16 |


### Key Findings
- XGBoost performed best with 92.31% accuracy
- High precision and recall across all models
- Effective classification of Parkinson's Disease using voice features

## Requirements
```bash
pip install -r requirements.txt
```

## Installation
```bash
git clone https://github.com/yourusername/parkinsons-voice-classification.git
cd parkinsons-voice-classification
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Acknowledgments
- Kaggle for the dataset
- Original researchers for data collection