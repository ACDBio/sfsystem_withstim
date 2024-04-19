from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import mne
import yasa
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_iris
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_iris # Example dataset
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import sys
from glob import glob
from tqdm.autonotebook import tqdm
import pickle
from joblib import dump, load
from hmmlearn import hmm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler






def train_sklearn_classifier(X=None, y=None, X_test=None, y_test=None, X_train=None, y_train=None, clf=LogisticRegression, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l2']}, 
              test_frac=0.5, 
              verb=True, 
              metric='roc_auc',
              refit=True,
              cv=2,
              use_rus=True,
             gridder='GridSearchCV'):
    
    if str(X)!='None':
        if use_rus:
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
        else:
            X_resampled, y_resampled = X, y
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_frac, random_state=42)
    else:
        print('Test split provided, not doing the split')
        if use_rus:
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train     
    
    if gridder=='GridSearchCV':
        grid_search = GridSearchCV(clf(), param_grid, cv=cv, scoring=metric, n_jobs=-1, refit=refit)
    else:
        grid_search = RandomizedSearchCV(clf(), param_grid, cv=cv, scoring=metric, n_jobs=-1, n_iter=10)
        print('using randomized search')
    grid_search.fit(X_train, y_train)
    y_pred_proba =  grid_search.best_estimator_.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
    diff = tpr - fpr
    optimal_idx = np.argmax(diff)
    optimal_threshold = thresholds[optimal_idx]



    y_pred = grid_search.best_estimator_.predict_proba(X_test)
    y_pred=y_pred[:,1]
    roc_auc = roc_auc_score(y_test, y_pred)

    y_pred=y_pred>optimal_threshold
    clsrep=classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    fig=plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    cv_results=grid_search.cv_results_

    if verb==True:
        #print('CV score:')
        #print(roc_auc)
        print('ROC-AUC:')
        print(roc_auc)
        print('Classification Report:')
        print(clsrep)
        print('Confusion Matrix:')
        print(cm)
        plt.show()

    return grid_search.best_estimator_, roc_auc, clsrep, cm, fig, optimal_threshold, cv_results


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
l2_penalty_grid = {
    'penalty': ['l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # Includes 'liblinear'
    'max_iter': [500, 10000]
}
l1_penalty_grid = {
    'penalty': ['l1'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['sag', 'saga'], # Excludes 'liblinear'
    'max_iter': [500, 10000]
}

classifiers_and_grids = [
    {
        'classifier': LogisticRegression,
        'grid': l1_penalty_grid,
        'name':'LogisticRegression_l1'

    },
     {
        'classifier': LogisticRegression,
        'grid': l2_penalty_grid,
        'name':'LogisticRegression_l2'

    },
    {
        'classifier': DecisionTreeClassifier,
        'grid': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]},
        'name':'DecisionTreeClassifier'
    },
    {
        'classifier': RandomForestClassifier,
        'grid': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'name':'RandomForestClassifier'
    },
    {
        'classifier': SVC,
        'grid': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4, 5],
            'gamma': ['scale', 'auto']
        },
        'name':'SVC'
    },
    {
        'classifier': KNeighborsClassifier,
        'grid': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'name':'KNeighborsClassifier'
    },
    {
        'classifier': GradientBoostingClassifier,
        'grid': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.5, 0.75, 1.0],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'name':'GradientBoostingClassifier'
    },
    {
        'classifier': AdaBoostClassifier,
        'grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'name':'AdaBoostClassifier'
    },
    {
        'classifier': GaussianNB,
        'grid': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        },
        'name':'GaussianNB'
    },
    {
        'classifier': MLPClassifier,
        'grid': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        },
        'name':'MLPClassifier'
    }
]

n_points_pre_min=100
n_points_pre_max=350
window_init=150
n_samples=500
dataset='haagladen_20_slowwaves_rawframes'
with open(f'{dataset}_pmin_{n_points_pre_min}_pmax_{n_points_pre_max}_w_{window_init}_scalos_rsamples_{n_samples}.pkl', 'rb') as f:
    scalo_datas, scalo_labels=pickle.load(f)
trainx=np.vstack(scalo_datas[:15])
trainy=np.hstack(scalo_labels[:15])
testx=np.vstack(scalo_datas[15:]) #different people are used for training and testing!
testy=np.hstack(scalo_labels[15:])
cv=3
windows=[150] #, 100, 50]
n_feats=32 #n frequencies from the continuous transform
force=False
results={}

stand=StandardScaler()
scaler=MinMaxScaler()

transforms=[stand, scaler]

for transformer in transforms:
    transformer.fit(trainx)
    trainx=transformer.transform(trainx)
    testx=transformer.transform(testx)



for win in tqdm(windows):
    print(f'Processing window {win}')
    winlength=win*n_feats
    ctrx=trainx[:,-winlength:]
    ctsx=testx[:,-winlength:]
    for classdata in tqdm(classifiers_and_grids):
        classifier=classdata['classifier']
        grid=classdata['grid']
        name=classdata['name']
        print(name)
        if force==False:
            if os.path.exists(f'STATS_{name}_{dataset}_min_{n_points_pre_min}_max_{n_points_pre_max}_wi_{window_init}_wa_{win}_ns_{n_samples}.pkl'):
                proceed=False
                print('STATS File exists. Skipping.')
        else:
            proceed==True

        if proceed==True:
            try:
                best_estim, roc_auc, clsrep, cm, fig, optimal_threshold, cvres = train_sklearn_classifier(X_test=ctsx, y_test=testy, 
                                                                    X_train=ctrx, y_train=trainy, 
                                                                    clf=classifier, param_grid=grid, test_frac=0.5, verb=False, use_rus=False, cv=cv, refit=True, metric='roc_auc',
                                                                    gridder='GridSearchCV')
                print(roc_auc)
                print(clsrep)
                results[name]={'roc_auc':roc_auc, 'clsrep':clsrep, 'cm':cm, 'clsfile':f'CLASSIFIER_{name}_{dataset}_min_{n_points_pre_min}_max_{n_points_pre_max}_wi_{window_init}_wa_{win}_ns_{n_samples}_tstroc_{roc_auc}_thresh_{optimal_threshold}.joblib', 'cv': cvres}
                dump([transforms, best_estim], f'CLASSIFIER_{name}_{dataset}_min_{n_points_pre_min}_max_{n_points_pre_max}_wi_{window_init}_wa_{win}_ns_{n_samples}_tstroc_{roc_auc}.joblib')
                with open(f'STATS_{name}_{dataset}_min_{n_points_pre_min}_max_{n_points_pre_max}_wi_{window_init}_wa_{win}_ns_{n_samples}.pkl', 'wb') as file:
                    pickle.dump({'roc_auc':roc_auc, 'clsrep':clsrep, 'optimal_threshold':optimal_threshold,'cm':cm, 'cvres': cvres}, file)
                
            except Exception as e:
                print(e)
with open(f'CLASSIFIER_ALLSTATS_{dataset}_min_{n_points_pre_min}_max_{n_points_pre_max}_wi_{window_init}.pkl', 'wb') as file:
    pickle.dump(results, file)  