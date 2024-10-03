import mlflow
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pickle
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


PROJECT = os.environ['PROJECT']


def read_train_test(path):
    train = pd.read_csv(f'{PROJECT}/data/preprocessed/train_data.csv')
    test = pd.read_csv(f'{PROJECT}/data/preprocessed/test_data.csv')
    return train, test


def preprocess(train_test):
    train, test = train_test
    cat_columns = ['weather_now']
    numeric_columns = [column for column in train.columns if column != 'weather_now' and column != 'weather_next_hour']

    preprocessor = ColumnTransformer(
        [
            ('scaler', StandardScaler(), numeric_columns),
            ('encoder', OneHotEncoder(sparse_output=False), cat_columns)
        ]
    )
    preprocessor.set_output(transform='pandas')
    
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return (X_train_transformed, y_train, X_test_transformed, y_test), preprocessor


def train(data, preprocessor):
    # Enable autlogging
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("model weather pred")
    mlflow.autolog()
    with mlflow.start_run():
        X_train, y_train, X_test, y_test = data
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        params = {
            'max_depth': [5, 10],
            'max_features': ['sqrt', X_train.shape[1]],
            'n_estimators': [10, 100, 200],
            'class_weight': [{0:1, 1:10}, {0:1, 1:1}],
            'ccp_alpha': [0.01, 0]
        }

        rf_grid = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=cv, scoring='roc_auc', verbose=2)
        rf_grid.fit(X_train, y_train)
        best_score, best_treshold = 0, np.nan
        for t in np.linspace(0, 1, 100):
            y_pred = rf_grid.predict_proba(X_test)
            y_pred = y_pred[:, 1] > t
            if roc_auc_score(y_test, y_pred) > best_score:
                best_score = roc_auc_score(y_test, y_pred)
                best_treshold = t

        y_pred = rf_grid.predict_proba(X_test)
        y_pred = y_pred[:, 1] > best_treshold
        print("Best threshold: ", best_treshold)
        print(classification_report(y_test, y_pred))
        print(roc_auc_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        mlflow.log_param("best_threshold", best_treshold)
        mlflow.log_metric("best_score", best_score)
        pipeline = Pipeline(
        [('preprocess', preprocessor),
        ('predictor', rf_grid.best_estimator_)]
        )
        with open(f"{PROJECT}/models/weather_classifier.pkl", 'wb') as f:
            pickle.dump(pipeline, f) 
        
        
def main():
    train_test = read_train_test(f'{PROJECT}/data/preprocessed')
    data, preprocessor = preprocess(train_test)
    train(data, preprocessor)
    
    
if __name__ == '__main__':
    main()


    
    


