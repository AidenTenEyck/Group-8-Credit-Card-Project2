The goal of this project is to narrow down and accurately predict people who are at high risk for diabetes by creating an algorithm using the diabetes prediction dataset. 
  1. The dataset used has 100,000 rows and 9 columns. 
  2. The data is medical in nature and includes details like whether the person has smoked, has heart disease, bmi, HbA1c level, blood glucose level, hypertension, and if they have diabetes. 
  3. The data is highly imbalanced with few people having heart disease, hypertension, or even diabetes.

We used five classifiers to analyze the data and train the model such as Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, and KNeighborsClassifier. 
  1. The model is more than 80% accurate using any of the above classifiers. 
  2. The most accurate classifier is Random Forest which came in between 96-97% accuracy even when being undersampled or oversampled.
  3. Some of the findings are that a higher level of HbA1c- and higher blood glucose levels can serve as a strong indicator of diabetes. 
  4. There appears to be some correlation between a person's age and if they have diabetes which could indicate that as you get older you have a higher chance of getting diabetes.
  5. Based on key drivers(Balanced Accuracy Score, ROC AUC Scores, True Negatives (Minimization), Recall/F1-Scores) we have selected  random forest for the prediction model.

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

-Contains common fuctions to run all models for specified technique

  4. hyper_parameters_tuning.py:

-Determine best fit for K nearest Neighbors, max depth for Random Forest, and feature importance.
