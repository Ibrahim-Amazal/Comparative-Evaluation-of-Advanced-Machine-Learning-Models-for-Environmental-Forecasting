# Comparative Evaluation of Advanced Machine Learning Models in Forecasting Environmental Patterns

This repository presents a high-expertise comparison of Prophet, LSTM, SVR, TCN, and Transformer for forecasting temperature, wind turbine SCADA, and air quality data. Using CRISP-DM, it ensures robust preprocessing, model tuning, and interpretability (SHAP, RMSE, SMAPE) for real-world applications. 

📖 Research Context & Contribution
This work systematically applies and evaluates five state-of-the-art machine learning architectures for environmental forecasting across three diverse real-world datasets (temperature, wind turbine performance, and air quality monitoring). The study integrates an academic-grade methodology, following the CRISP-DM framework, and establishes a reproducible benchmarking pipeline for forecasting in dynamic, high-dimensional environmental systems.

## Overview

This repository presents a rigorous comparative evaluation of advanced machine learning models for forecasting environmental parameters such as temperature, wind turbine performance, and air quality. The study systematically analyzes five state-of-the-art forecasting models:

Prophet – Captures long-term seasonality via additive models.

- LSTM (Long Short-Term Memory) – Specialized in handling sequential and temporal dependencies.

- SVR (Support Vector Regression) – Effective in capturing non-linear relationships.

- TCN (Temporal Convolutional Network) – Designed for hierarchical temporal patterns.

- Transformer – Utilizes attention mechanisms to model multi-variable dependencies.

The project applies these models to three distinct datasets:

1. Temperature Dataset – Hourly temperature observations from Central-West Brazil.

2. Wind Turbine Dataset – SCADA data capturing wind turbine performance.

3. Air Quality Dataset – Daily pollutant concentration levels across Indian regions.

By following a structured methodology based on the CRISP-DM framework, this work emphasizes reproducibility, robust evaluation metrics, and insightful model comparisons.

## Project Structure

This repository is organized into the following directories and files:

├── data/                     # Datasets used in the project (zipped for storage efficiency)

├── notebooks/

│   ├── EDA/                  # Exploratory Data Analysis notebooks for each dataset

│   │   ├── EDA_Temperature_Analysis_Brazil.ipynb

│   │   ├── EDA_Wind_Turbine_Analysis.ipynb

│   │   ├── EDA_Air_Quality_Analysis.ipynb

│   ├── Models/               # Machine learning model implementation notebooks

│   │   ├── ML_Model_Implementation_On_Temperature_Dataset.ipynb

│   │   ├── ML_Model_Implementation_On_Wind_Turbine_Dataset.ipynb

│   │   ├── ML_Model_Implementation_On_Air_Quality_Dataset.ipynb

├── src/                      # Utility scripts for data preprocessing

│   ├── data_preprocessing_utils.ipynb

├── docs/                     # Project documentation, including research reports

│   ├── ML_Environmental_Forecasting_Report.docx

├── requirements.txt          # Dependencies and required libraries for running the project

├── README.md                 # This file - Project overview and repository navigation guide

## Installation & Setup

To reproduce the results or experiment with the models, follow these steps:

1. Clone this repository

git clone https://github.com/yourusername/Comparative_Evaluation_ML_Environmental_Forecasting.git
cd Comparative_Evaluation_ML_Environmental_Forecasting

2. Set up the environment (Recommended: Use a virtual environment)

python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows

3. Install dependencies

pip install -r requirements.txt

4. Run Jupyter Notebooks or any other alternative

jupyter notebook

Navigate to notebooks/EDA/ for data exploration or notebooks/Models/ to run model training.

## Project Requirements

To ensure smooth execution, install the following dependencies listed in requirements.txt:
=> numpy

=> pandas

=> matplotlib

=> seaborn

=> scikit-learn

=> statsmodels

=> prophet

=> tensorflow

=> keras

=> torch

=> torchaudio

=> torchvision

=> xgboost

=> shap

These packages ensure seamless execution of the models, preprocessing utilities, and visualization tools.

## Key Features

Comprehensive Exploratory Data Analysis (EDA): Each dataset undergoes a thorough investigation for missing values, trends, and feature correlations.

Reusable Data Preprocessing Utilities: A structured pipeline for feature engineering, scaling, and handling missing values.

Robust Model Evaluation: RMSE, SMAPE, SHAP analysis, and coherence plots are utilized for deep model interpretability.

Real-World Applicability: The project addresses forecasting challenges in agriculture, renewable energy, and pollution monitoring.

## Results

Temperature Dataset: TCN demonstrated the best performance in capturing periodic dependencies.

Wind Turbine Dataset: LSTM effectively modeled both short- and long-term variations in turbine dynamics.

Air Quality Dataset: Transformer excelled in handling multi-variable interactions and capturing pollutant dependencies.

The full model evaluation and insights are available in the research document:
📄 ML_Environmental_Forecasting_Report.docx

## Future Work

Hybrid model approaches: Combining TCN and Transformer models to leverage their complementary strengths.

Automated Hyperparameter Tuning: Bayesian optimization for refining model configurations.

Expansion to Larger Datasets: Increasing the scope with real-time, high-resolution environmental data.

## Contact

📧 Email: IbrahimAmazal2024@gmail.com🔗 LinkedIn: (https://www.linkedin.com/in/ibrahim-amazal-5a7875132/)

For further inquiries or contributions, feel free to open an issue or submit a pull request!

## Acknowledgments

This project was developed as part of an advanced machine learning study at the University of Limerick and integrates diverse data sources for a high-impact comparative analysis. Special thanks to contributors and research peers for their invaluable feedback.

