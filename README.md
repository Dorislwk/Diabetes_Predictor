# Diabetes Analysis and Predictor

## 1. Introduction 🎯
The diabetes predictor is an informational tool for analyzing various demographic data and health metrics related to diabetes, alerting users with health risk messages.

## 2. Table of Contents 📔 
- Core Features
- Project Structure
- Files and Usage
- Getting Started
- GUI Overview
- Acknowledgments

## 3. Core Features
1. Analysis various factor interactive correlation with diabetes
2. Predictive modeling of diabetes status with logistic regression
3. User interface for diabetes risk prediction
4. Database

## 4. Project Structure

```
│
├── diabetes_balanced_patients.csv
├── diabetes_correlation_analysis.py 🚀
└── model_training_predictor.py

```

## 5. Files and Usage 🔍
- diabetes_balanced_patients.csv: A combined dataset for analysis, training, and testing.

- diabetes_correlation_analysis.py: Handles data cleaning and processing steps and functions for analyzing diabetes

- model_training_predictor.py: Contains functions for preparing data for training the logistic regression model, visualizing analysis, predictor GUI and connecting to SQLite.

## 6. Getting Started 🧰
1. Setup: Ensure you have the required libraries installed. You can install them using pip:

```
pip install -r requirements.txt
```

2. Data Processing and Analysis: Execute the diabetes_correlation_analysis.py file to start the analysis:

```
python diabetes_correlation_analysis.py
```


3. Model Training, Testing and Predictor GUI: Select features for model training, visualize graph and create a predictor using Tkinter.

```
model_training_predictor.py
```
## 7. GUI Overview

### Main View
There is a prediction interface powered by Tkinter that provides the main functions for analysis and prediction. Here is the preview of GUI.

![image](https://github.com/Dorislwk/Diabetes_Predictor/blob/main/Photo/Predictor_interface.png)

![image](https://github.com/Dorislwk/Diabetes_Predictor/blob/main/Photo/question_example.png)

There are total 21 questions and should be filled in by numbers.


![image](https://github.com/Dorislwk/Diabetes_Predictor/blob/main/Photo/result_message.png)


![image](https://github.com/Dorislwk/Diabetes_Predictor/blob/main/Photo/disclaimer.png)

### Database
![image](https://github.com/Dorislwk/Diabetes_Predictor/blob/main/Photo/database_firsthalf.png)

![image](https://github.com/Dorislwk/Diabetes_Predictor/blob/main/Photo/database_secondhalf.png)

## 8. Acknowledgments 📊
-**Data Sources** : Diabetes analysis and prediction dataset is from https://www.kaggle.com/code/rahul713/diabetes-data-analysis#Diabetes_predictor

