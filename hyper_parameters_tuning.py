##--------------------------------------------------------------------
""" 
Import Dependencies
"""
##--------------------------------------------------------------------
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
import matplotlib.pyplot as plt
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')

##--------------------------------------------------------------------
"""
Model features evaluates the following:

Runs K Nearest Neighbors with k: 1 - 20 step 2  to determined best fit
Runs Random Forest to determoine best max_depth with max depth 1 - 40, step 2
Runs feature_importances_ to showcase the significant features that affect diabetes prediction

"""
##--------------------------------------------------------------------

# COMMAND ----------

def hyper_parameter_tuning(data,target):
    
    label_encoders = {}
    for column in ['gender', 'smoking_history']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    data.head(5)

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Loop through different k values to find which has the highest accuracy.
    # Note: We use only odd numbers because we don't want any ties.
    train_scores = []
    test_scores = []
    for k in range(1, 20, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
    
    # Plot the results
    plt.plot(range(1, 20, 2), train_scores, marker='o', label="training scores")
    plt.plot(range(1, 20, 2), test_scores, marker="x", label="testing scores")
    plt.xlabel("k neighbors")
    plt.ylabel("accuracy score")
    plt.legend()
    plt.show()
    print("-" * 100)

    # Create a loop to vary the max_depth parameter
    # Make sure to record the train and test scores 
    # for each pass.

    # Depths should span from 1 up to 40 in steps of 2
    depths = range(1, 40, 2)

    # The scores dataframe will hold depths and scores
    # to make plotting easy
    scores = {'train': [], 'test': [], 'depth': []}

    # Loop through each depth (this will take time to run)
    for depth in depths:
        clf = RandomForestClassifier(max_depth=depth)
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        scores['depth'].append(depth)
        scores['train'].append(train_score)
        scores['test'].append(test_score)
    print("-" * 100)
    
    ##@@
    scores_pl = pd.DataFrame(scores)
    scores_pl.plot(x="depth")
    ##@@

    
    # Create a dataframe from the scores dictionary and
    # set the index to depth
    scores_df = pd.DataFrame(scores).set_index('depth')
    print(scores_df)

    print("")
    print("-" * 100)
    print("")

    rf_model = RandomForestClassifier(n_estimators=128, random_state=78)
    rf_model = rf_model.fit(X_train, y_train)
    importances = rf_model.feature_importances_
    # List the top 10 most important features
    importances_sorted = sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
    print(importances_sorted[:15])