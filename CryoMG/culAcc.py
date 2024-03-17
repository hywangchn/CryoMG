import numpy as np
import copy
from sklearn import svm,metrics
from sklearn.model_selection import cross_validate,train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, fbeta_score, balanced_accuracy_score, confusion_matrix

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

def cross_val_confusion_matrix(clf,X,y,skf):
    if type(y) is list:
        y = np.array(y)
    c_matrix = []
    pre_label_record = None
    pre_index_record = None
    for train_index, test_index in skf.split(X, y):
        clf.fit(X[train_index],y[train_index])
        pre_label = clf.predict(X[test_index])
        if pre_index_record is None:
            pre_index_record = copy.deepcopy(test_index)
            pre_label_record = copy.deepcopy(pre_label)
        else:
            pre_index_record = np.concatenate([pre_index_record,test_index], axis=0)
            pre_label_record = np.concatenate([pre_label_record,pre_label], axis=0)
        matrix = confusion_matrix(y[test_index],pre_label).ravel()
        c_matrix.append(matrix.tolist())
    return np.array(c_matrix), np.array([pre_index_record, pre_label_record]).transpose()

def SKFold_accuracy_cross_Grid(X, y):
    svmlabel = copy.deepcopy(y)
    svmlabel[svmlabel==0]=-1
    #base parameter
    if X.shape[0]>=20:
        nsplit = 10
    else:
        nsplit = 5
    scoring_fnc = make_scorer(confusion_matrix)
    cv = StratifiedKFold(n_splits=nsplit, random_state=None)
    
    clf_knn = KNeighborsClassifier()
    clf_bayes = naive_bayes.GaussianNB()
    clf_dtree = DecisionTreeClassifier()
    clf_xgb = xgb.XGBClassifier()
    clf_lgb = lgb.LGBMClassifier()
    
    acc_knc, pre_label_knn = cross_val_confusion_matrix(clf_knn, X, y, skf=cv)
    acc_nb, pre_label_nb = cross_val_confusion_matrix(clf_bayes, X, y, skf=cv)
    acc_dt, pre_label_dt = cross_val_confusion_matrix(clf_dtree, X, y, skf=cv)
    acc_xgb, pre_label_xgb = cross_val_confusion_matrix(clf_xgb, X, y, skf=cv)
    acc_lgb, pre_label_lgb = cross_val_confusion_matrix(clf_lgb, X, y, skf=cv)

    return {"knn":acc_knc, "bayes":acc_nb, "dtree":acc_dt, "xgb":acc_xgb, "lgb":acc_lgb}, {"knn":pre_label_knn, "bayes":pre_label_nb, "dtree":pre_label_dt, "xgb":pre_label_xgb, "lgb":pre_label_lgb}
