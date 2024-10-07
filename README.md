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
     
This algorithm could be used as an early warning system in a hospital to quickly narrow down patients who are at high risk for diabetes, or alternatively it could be used by medication companies by helping them narrow down their target audience to increase the effectiveness of advertisements for their products.
