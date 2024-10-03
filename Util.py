# Databricks notebook source
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTEN
import matplotlib.pyplot as plt


# COMMAND ----------

def model_evaluation(data,target,sample_method):
    models_to_run = [RandomForestClassifier(), LogisticRegression(),DecisionTreeClassifier(),GradientBoostingClassifier(),KNeighborsClassifier()]

    label_encoders = {}
    for column in ['gender', 'smoking_history']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    data.head(5)

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    sampling_methods = {
        "RandomUnderSampler": RandomUnderSampler(),
        "RandomOverSampler": RandomOverSampler(),
        "SMOTE": SMOTE(),
        "SMOTEN": SMOTEN()
    }
    
    # Check if the sample_method is valid and then apply the corresponding sampling technique
    if sample_method in sampling_methods:
        sampler = sampling_methods[sample_method]
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        
    for model in models_to_run:
        print(f"{model.__class__.__name__} {sample_method} Report:")

        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy of {model.__class__.__name__}: {accuracy}")

        print('Train Model Score: %.3f' % pipeline.score(X_train, y_train))
        print('Test Model Score: %.3f' % pipeline.score(X_test, y_test))

        print('Balanced Train Accuracy Score:', balanced_accuracy_score(y_train, y_train_pred))
        print('Balanced Test Accuracy Score:', balanced_accuracy_score(y_test, y_pred))

        pred_probas = pipeline.predict_proba(X_test)
        pred_probas_firsts = [prob[1] for prob in pred_probas]
        print('Test roc_auc_score', roc_auc_score(y_test, pred_probas_firsts))

        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

        print(f"{model.__class__.__name__} {sample_method} Classification Report:")
        print(classification_report(y_test, y_pred))

        print("**********************************************************************")
