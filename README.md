Objective

The goal of this project is to narrow down and accurately predict people who are at high risk for diabetes by creating an algorithm using the diabetes prediction dataset. 
  1. The dataset used has 100,000 rows and 9 columns. 
  2. The data is medical in nature and includes details like whether the person has smoked, has heart disease, bmi, HbA1c level, blood glucose level, hypertension, and if they have diabetes. 
  3. The data is highly imbalanced with few people having heart disease, hypertension, or even diabetes.

Details about the dataset
The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of patient their BMI, , age, blood_glucose_level and so on.

Key Features
Data Collection and Processing: 
The project involves collecting a dataset containing features related to individuals' health, such as glucose levels, blood pressure, BMI, and more. Using Pandas, the collected data is cleaned, preprocessed, and transformed to ensure it is suitable for analysis. The dataset is included in the repository for easy access.

Data Visualization: 
The project utilizes data visualization techniques to gain insights into the dataset. By employing profile reporting and correlation matrices are created. These visualizations provide a deeper understanding of the data distribution and relationships between features.

Train-Test Split: To evaluate the performance of the classification model, the project employs the train-test split technique. The dataset is divided into training and testing subsets, ensuring that the model is trained on a portion of the data and evaluated on unseen data. This allows for an accurate assessment of the model's ability to generalize to new data.

Feature Scaling: As part of the preprocessing pipeline, the project utilizes the StandardScaler from Scikit-learn to standardize the feature values. Standardization ensures that all features have a mean of 0 and a standard deviation of 1, which can help improve the performance and convergence of certain machine learning algorithms.


Supervised Machine Learning Algorithm.
We used five classifiers to analyze the data and train the model such as Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, and KNeighborsClassifier. 
  1. The model is more than 80% accurate using any of the above classifiers. 
  2. The most accurate classifier is Random Forest which came in between 96-97% accuracy even when being undersampled or oversampled.
  3. Some of the findings are that a higher level of HbA1c- and higher blood glucose levels can serve as a strong indicator of diabetes. 
  4. There appears to be some correlation between a person's age and if they have diabetes which could indicate that as you get older you have a higher chance of getting diabetes.
  5. Based on key drivers(Balanced Accuracy Score, ROC AUC Scores, True Negatives (Minimization), Recall/F1-Scores) we have selected  random forest for the prediction model.

Result

Classification Report:
  1. Class 0: (Non-diabetes): 
  The model has a high precision (0.97) for class 0, meaning that among all instances where the model predicted non-diabetes, 97% were indeed non-diabetes.
  The recall for class 0 is also high (1.0). This means that our model correctly identified 100% of all actual non-diabetes cases in the dataset.
  2. Class 1 (Diabetes): 
  The precision for class 1 is lower around (0.95), which indicates that when the model predicted diabetes, it was correct around 95% of the time.

This algorithm could be used as an early warning system in a hospital to quickly narrow down patients who are at high risk for diabetes, or alternatively it could be used by medication companies by helping them narrow down their target audience to increase the effectiveness of advertisements for their products.

Files:
  1. ML_Project2.ipynb:

-Jupyter notebook containing models and graphs to do data analysis

  3. Util.py: 

-Contains common fuctions  and pipeline to run all models for specified technique

  4. hyper_parameters_tuning.py:

-Determine best fit for K nearest Neighbors, max depth for Random Forest, and feature importance.
