##--------------------------------------------------------------------

"""
Import Dependencies
"""
##--------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
from imblearn.combine import SMOTEENN

import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

##--------------------------------------------------------------------
"""
Project Scope: Problem being solved is detection of diabetes in patient. This is a. Binary Classification problem.

Description: The below utility function employs a pipeline to run 5 different models against an input data set
Models evaluated: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, GradientBoostingClassifier, KNeighborsClassifier

Inputs: DataSet dataframes - Original dataset, Random Under Sampled dataset, Random Over Sampled dataset, SMOTE Over Sampled dataset, SMOTEENN Combined Dataset
Target dataframe - diabetes prediction
Type of dataframe - indicating which technique is being uutilized

Outputs: Metrics/Scores, Confusion Matrix, Classification Report for each model evaluated

"""
##--------------------------------------------------------------------
def model_evaluation(data,target,sample_method):
    
    #Define the list of models to evaluate
    models_to_run = [
        RandomForestClassifier(),
        LogisticRegression(),
        DecisionTreeClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier()
    ]

    data = data.drop_duplicates()
    
    #Create label encodes for categoricals:
    label_encoders = {}
    for column in ['gender', 'smoking_history']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    data.head(5)

    # Create the datasets X, y, train/test splits
    
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    #Compensate for highly imbalanced dataset using over sampling and undersampling techniques
    sampling_methods = {
        "RandomUnderSampler": RandomUnderSampler(),
        "RandomOverSampler": RandomOverSampler(),
        "SMOTE": SMOTE(),
        "SMOTEENN":SMOTEENN()
    }
    
    scores = {'Model': [], 'Dataset Used': [], 'Testing Model Score': [], \
                  'Training Model Score': [], 'Accuracy Score (Testing)': [], 'Balanced Accuracy Score (Testing)': [], \
                  'Balanced Accuracy Score (Training)': [], 'ROC AUC Score (Testing)': [] }
    
    # Check if the sample_method is valid and then apply the corresponding sampling technique
    if sample_method in sampling_methods:
        sampler = sampling_methods[sample_method]
        X_train, y_train = sampler.fit_resample(X_train, y_train)
            
    #For each mode in the list, run the model and determine the model scores
    for model in models_to_run:
        print("*" * 100)
        
        print(f"{model.__class__.__name__} {sample_method} Report:")
        
        ### Below works
        print("-" * 50)
        print("Original Data Value Counts")
        print(y.value_counts())
        print("-" * 50)
        print("Training Data Value Counts")
        print(y_train.value_counts())
        print("-" * 50)
        print("Training Data Value Counts, by Pct")
        print(y_train.value_counts(normalize=True)*100)
        print("-" * 50)
        #### SRI
        
        sample_value_counts_df = pd.DataFrame(y.value_counts())
        sample_value_counts_df.to_csv("eda/" + sample_method + "_" + str(model.__class__.__name__) + "_" + "sample_dist.csv", index=True) 
    
        training_data_value_counts_df = pd.DataFrame(y_train.value_counts())
        training_data_value_counts_df.to_csv("eda/" + sample_method + "_" + str(model.__class__.__name__) + "_" + "training_dist.csv", index=True)
        
        training_pct_value_counts_df = pd.DataFrame(y_train.value_counts(normalize=True)*100)
        training_pct_value_counts_df.to_csv("eda/" + sample_method + "_" + str(model.__class__.__name__) + "_"  + "training_pct_dist.csv", index=True)
    
        #Create a pipeline
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)

        #Evaluate model scores
        accuracy = round(accuracy_score(y_test, y_pred),2)
        print(f'Accuracy of {model.__class__.__name__}: {accuracy}')

        testing_model_score = round(pipeline.score(X_test, y_test),2)
        training_model_score = round(pipeline.score(X_train, y_train),2)
        print(f'Testing Model Score: {testing_model_score}')
        print(f'Training Model Score: {training_model_score}')
                
        #print('Train Model Score: %.3f' % pipeline.score(X_train, y_train))
        #print('Test Model Score: %.3f' % pipeline.score(X_test, y_test))
        
        balanced_accuracy_score_testing = round(balanced_accuracy_score(y_test, y_pred),2)
        balanced_accuracy_score_training = round(balanced_accuracy_score(y_train, y_train_pred),2)
              
        print(f'Balanced Testing Accuracy Score: {balanced_accuracy_score_testing}')
        print(f'Balanced Training Accuracy Score: {balanced_accuracy_score_training}')

        pred_probas = pipeline.predict_proba(X_test)
        pred_probas_firsts = [prob[1] for prob in pred_probas]
        print(f'Test ROC AUC Score: {roc_auc_score(y_test, pred_probas_firsts)}')

        scores['Model'].append(model.__class__.__name__)
        scores['Dataset Used'].append(sample_method)
        scores['Testing Model Score'].append(testing_model_score)
        scores['Training Model Score'].append(training_model_score)
        scores['Accuracy Score (Testing)'].append(accuracy)
        scores['Balanced Accuracy Score (Testing)'].append(balanced_accuracy_score_testing)
        scores['Balanced Accuracy Score (Training)'].append(balanced_accuracy_score_training)
        scores['ROC AUC Score (Testing)'].append(pred_probas_firsts)       
        
        #get confusion matrix as it is a binary classification
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
        cm_display.plot()
        plt.show()

        print(f"{model.__class__.__name__} {sample_method} Classification Report:")
        print(classification_report(y_test, y_pred))

        print("*" * 100)
        
    # Create a dataframe from the scores dictionary and
    # set the index to depth
    scores_df = pd.DataFrame(scores)
    #print("-" * 100)
    #print("Evaluation Across Models")
    #print(scores_df)
    #print("-" * 100)
    scores_df.set_index('Model', inplace=True)
    scores_df.to_csv("output/" + sample_method + "_" + "Scores.csv", index=True)
        
        ######


    
