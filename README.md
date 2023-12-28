# Classification with K-Nearest Neighbors

**Lab by:** Thomas Haskell

## Description

This lab explores the application of the K-Nearest Neighbors (KNN) algorithm for classification, using customer demographic data. The goal is to classify telecommunications customers into specific identification groups based on features such as region, tenure, age, and income. The lab is part of the IBM Machine Learning with Python Certification Course.

## Objectives

1. Utilize K-Nearest Neighbors to classify a customer data set.
2. Understand how data points with similar features can help classify unknown observations.

## Data Source

The dataset used in this lab can be found [here](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv).

## Prerequisites

- Python 3.x
- numpy, matplotlib, pandas, scikit-learn

## Getting Started

1. Clone the repository or download the zip file from GitHub.
2. Run the provided Python script to download the dataset.
3. Execute the lab script (`KNN_Classification_Lab.py`).

## Exploratory Data Analysis

- View the first few rows of the dataset.
- Explore the distribution of the target variable (`custcat`) using histograms.

## Feature Selection

- Select relevant features for classification.
- Normalize the data for optimal model performance.

## Model Training and Evaluation

- Split the dataset into training and testing sets.
- Implement the K-Nearest Neighbors algorithm with different numbers of neighbors.
- Evaluate model accuracy on both training and testing sets.

## Results

- Compare the accuracy of KNN models with varying numbers of neighbors.
- Visualize the accuracy results to identify the optimal number of neighbors.

## Conclusion

This lab provides hands-on experience with K-Nearest Neighbors for classification tasks, offering insights into the impact of feature selection and the choice of the number of neighbors on model performance.
