# Diabetes Prediction Using Machine Learning

This repository contains an implementation of a machine learning project to predict diabetes status based on health indicators. The dataset used includes 21 features and a target variable with three classes:

* **0**: Non-diabetic
* **1**: Pre-diabetic
* **2**: Diabetic

The goal of this project is to classify individuals into one of these three categories using machine learning models and evaluate their performance using metrics like F1-macro scores.

## Features

* **HighBP**: High blood pressure
* **HighChol**: High cholesterol
* **CholCheck**: Cholesterol check
* **BMI**: Body Mass Index
* **Smoker**: Smoking status
* **Stroke**: History of stroke
* **HeartDiseaseorAttack**: History of heart disease or attack
* **PhysActivity**: Physical activity
* **Fruits**: Fruit consumption
* **Veggies**: Vegetable consumption
* **HvyAlcoholConsump**: Heavy alcohol consumption
* **AnyHealthcare**: Access to healthcare
* **NoDocbcCost**: Unable to see a doctor due to cost
* **GenHlth**: General health condition
* **MentHlth**: Mental health condition
* **PhysHlth**: Physical health condition
* **DiffWalk**: Difficulty walking/climbing stairs
* **Sex**: Gender
* **Age**: Age
* **Education**: Education level
* **Income**: Income level

## Project Overview

The project was conducted in the following stages:

1. **Data Analysis**:Analyzed the distribution of data and identified imbalances in target class distribution.
  
2. **Preprocessing**:
  
  * Rounded floating-point values to integers.
  * Capped BMI values greater than 50 to reduce outliers' influence.
  * Applied oversampling techniques to balance the dataset.
3. **Model Selection**:Compared multiple machine learning algorithms:
  
  * **Random Forest Classifier (RFC)**: Achieved the highest F1-macro score of 0.761.
  * **K-Nearest Neighbors (KNN)**: F1-macro score of 0.717.
  * **Logistic Regression** and **PCA-based models** were less effective.
4. **Hyperparameter Tuning**:
  
  * Adjusted `n_neighbors` for KNN and tree parameters for RFC.
  * Applied targeted oversampling for underrepresented classes.

## Results

| Model | F1-macro Score | Class 0 (F1) | Class 1 (F1) | Class 2 (F1) |
| --- | --- | --- | --- | --- |
| Random Forest | 0.761 | 0.95 | 0.65 | 0.69 |
| KNN | 0.717 | 0.94 | 0.56 | 0.65 |

## Files in the Repository

* **data.csv**: The dataset used in this project. (Since The file is too large, it's splited into 2 parts here.)
* **\*.py & \*.ipy**: Python codes for data exploration, preprocessing, and model training.
* **README.md**: This README file.

## Future Improvements

* Enhance feature engineering to capture more complex relationships between variables.
* Experiment with deep learning models like Neural Networks for improved performance.
* Apply advanced sampling techniques (e.g., SMOTE) to address class imbalances more effectively.
* Perform hyperparameter tuning using grid search or Bayesian optimization for better model fine-tuning.
