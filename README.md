# Failure Prediction using Machine Learning

This repository contains a Python code for predicting failures using machine learning techniques. It includes data preprocessing, exploratory data analysis (EDA), feature selection, model training, hyperparameter tuning, and evaluation.

## Dataset

The dataset used for this project is stored in the `failure.csv` file. It contains the following columns:

- `attribute1`, `attribute2`, `attribute3`, `attribute4`, `attribute5`, `attribute6`: Numerical features.
- `date`: Date of data collection.
- `device`: Device identifier.
- `failure`: Target variable (0 for non-failure, 1 for failure).

## Exploratory Data Analysis (EDA)

The EDA process involves the following steps:

- Summarizing the dataset, including the number of rows and columns, data types, and missing values.
- Visualizing the distribution of the target variable with respect to other features.
- Plotting pair plots and histograms for selected attributes.
- Visualizing the failure over time by month and week.
- Creating a correlation matrix to understand feature relationships.

## Data Preprocessing

Data preprocessing steps include:

- Converting the `date` column to a datetime format.
- Extracting additional date-related features such as `month`, `week`, `day_of_week`, `day_of_month`, and `is_weekend`.
- Handling class imbalance by using random under-sampling.

## Feature Selection

Feature selection involves identifying and selecting the most important features using machine learning models (e.g., RandomForest). The selected features are used for model training.

## Model Selection and Hyperparameter Tuning

The code includes the following machine learning models:

- GradientBoosting
- RandomForest
- AdaBoost
- ExtraTrees
- DecisionTree

Hyperparameter tuning is performed using the Optuna library. For models with an accuracy greater than 70%, hyperparameters are optimized to achieve the best results.

## Model Evaluation

Model evaluation includes metrics such as accuracy, precision, recall, and F1-score. Classification reports and confusion matrices are generated to assess model performance.

## Visualizations

The code includes various visualizations, such as pair plots, histograms, line plots, and heatmaps, to better understand the data and model performance.

## Prerequisites

- Python 3.x
- Required libraries: pandas, matplotlib, numpy, seaborn, scikit-learn, optuna, imbalanced-learn.
