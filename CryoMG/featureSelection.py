<<<<<<< HEAD
import numpy as np
import copy
import time
from collections import Counter
from scipy import stats
from minepy import MINE
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.model_selection import cross_validate,train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import naive_bayes
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, fbeta_score, balanced_accuracy_score, confusion_matrix
from culAcc import cross_val_confusion_matrix
import warnings
warnings.filterwarnings("ignore")

"""
filter
"""
def ttest(X,y):
    p_value = []
    p_label = [k for k in range(len(y)) if y[k]==0]
    n_label = [k for k in range(len(y)) if y[k]!=0]
    for i in range(len(X[0])):
        t, p = stats.ttest_ind(
                X[p_label, i],
                X[n_label, i],
                equal_var=False)
        p_value.append(p)
    ttest_index = np.argsort(np.array(p_value))
    return ttest_index,np.array(p_value)

def wtest(X,y):
    y[y==-1]=0
    p_value = []
    p_label = [k for k in range(len(y)) if y[k]==0]
    n_label = [k for k in range(len(y)) if y[k]!=0]
    for i in range(len(X[0])):
        t, p = stats.mannwhitneyu(
                X[p_label, i],
                X[n_label, i],
                alternative="two-sided")
        p_value.append(p)
    ttest_index = np.argsort(np.array(p_value))
    return ttest_index,np.array(p_value)

def pearson(X, y):
    pearson_value = []
    for i in range(len(X[0])):
        r, p = stats.pearsonr(X[:, i], y)
        pearson_value.append(p)
    pearson_index = np.argsort(np.array(pearson_value))
    return pearson_index,np.array(pearson_value)

def spearman_test(X, y):
    spearman_value=[]
    for i in range(X.shape[1]):
        spearman_value.append(abs(stats.spearmanr(X[:,i],y)[0]))
    spearman_value_index = np.argsort(np.array(spearman_value))
    return spearman_value_index, np.array(spearman_value)

def MICoe(X, y, r=0.3, alpha=0.6, c=15):
    y[y==-1]=0
    mine = MINE(alpha=0.6, c=15)
    micFC = []
    Subset = []
    for i in range(X.shape[1]):
        mic = mine.compute_score(X[:,i],y)
        micFC.append(mine.mic())
    Subset = np.argsort(-np.array(micFC))
    return Subset

def McOne(X, y, r=0.2, alpha=0.6, c=15):
    y[y==-1]=0
    mine = MINE(alpha=0.6, c=15)
    micFC = []
    Subset = []
    Selected = []
    for i in range(X.shape[1]):
        mic = mine.compute_score(X[:,i],y)
        micFC.append(mine.mic())
    Subset = np.argsort(-np.array(micFC))
    e=0
    while(e<len(Subset[np.where(np.array(micFC)>=r)])):
        q = e+1
        while(q<len(Subset[Subset>=r])):
            mine.compute_score(X[:,Subset[q]],X[:,Subset[e]])
            if mine.mic()>=micFC[Subset[q]]:
                q+=1
            else:
                break
        Selected.append(Subset[e])
        e = q
    mine.compute_score(X[:,len(Subset[Subset>=r])-2],X[:,len(Subset[Subset>=r])-1])
    if mine.mic()<micFC[Subset[-1]]:
        Selected.append(Subset[-1])
    if len(Selected)>100:
        Selected = Selected[:100]
    return Selected

def McTwo(X,y,feanum):
    y[y==-1]=0
    McOneIndex = McOne(X,y)
    if X.shape[0]>=20:
        nsplit = 10
    else:
        nsplit = 5
    SelectedFeatureIndex = BFS(X[:,McOneIndex],y,"knn",fea_num=feanum,nsplit=nsplit)
    return np.array(McOneIndex)[SelectedFeatureIndex]

def BFS(X, y, clf, fea_num=30, nsplit=10, random_state=None):
    svmlabel = copy.deepcopy(y)
    svmlabel[svmlabel==0]=-1
    #base parameter
    if X.shape[0]<20:
        nsplit = 5
    scoring_fnc = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=nsplit, random_state=None)
    #clissifier
    #svm
    tuned_parameters_svc = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        #{'kernel': ['poly'] ,'degree':[10,11,12,13]}
                        ]
    clf_svc_grid = GridSearchCV(SVC(probability=True,random_state=None), tuned_parameters_svc,scoring=scoring_fnc)
    #knn
    tuned_parameters_knn = [
        # {
        #     'weights':['uniform'],
        #     'n_neighbors':[i for i in [3,5]]
        # },
        {
            'weights':['distance'],
            'n_neighbors':[i for i in [3,5]],
            #'p':[i for i in range(1,6)]
        }
    ]
    clf_knn_grid = GridSearchCV(KNeighborsClassifier(),tuned_parameters_knn,scoring=scoring_fnc)
    #bayes
    clf_bayes_grid = naive_bayes.GaussianNB()
    #Dtree
    tuned_parameters_dtree = {'max_depth':range(1,5),'criterion':np.array(['entropy','gini'])}
    clf_dtree_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), tuned_parameters_dtree,scoring=scoring_fnc)
    if clf=="svm":
        ccc = clf_svc_grid
    elif clf=="knn":
        ccc = clf_knn_grid
    elif clf=="bayes":
        ccc = clf_bayes_grid
    elif clf=="dtree":
        ccc = clf_dtree_grid
    else:
        raise AssertionError()
    ########################################################
    SelectedFeatureIndex = []
    FeatureIndex = [i for i in range(X.shape[1])]
    for i in range(min(fea_num,X.shape[1])):
        max_index = None
        max_acc = 0
        for j in FeatureIndex:
            # t1 = time.time()
            acc = np.mean(cross_val_score(ccc,X[:,SelectedFeatureIndex+[j]],y))
            # print("time is : ",time.time()-t1)
            # print(acc)
            if acc>max_acc:
                max_acc = acc
                max_index = j
        # try:
        if max_index==None:
            print("pause")
        SelectedFeatureIndex.append(max_index)
        FeatureIndex.remove(max_index)
        # except:
        #     print(X.shape)
        #     print(max_index)
        #     raise AssertionError()
    return SelectedFeatureIndex


"""
feature-select and cul acc
"""

def TRankAcc(classifier, X, y, feature_num):
    if X.shape[0]<20:
        nsplit = 5
    else:
        nsplit = 10
    cv  = StratifiedKFold(n_splits=nsplit)
    argsort_index, _ = ttest(X,y)
    TrankIndex = BFS(X[:,argsort_index], y ,"knn", fea_num=feature_num, nsplit=nsplit)
    argsort_index = np.array(argsort_index)[TrankIndex]
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    print("accuracy is : ",scores)
    return classifier, argsort_index[:feature_num], scores, c_matrix

def WRankAcc(classifier, X, y, feature_num):
    if X.shape[0]<20:
        nsplit = 5
    else:
        nsplit = 10
    cv  = StratifiedKFold(n_splits=nsplit)
    argsort_index, _ = wtest(X,y)
    TrankIndex = BFS(X[:,argsort_index], y ,"knn", fea_num=feature_num, nsplit=nsplit)
    argsort_index = np.array(argsort_index)[TrankIndex]
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    print("accuracy is : ",scores)
    return classifier, argsort_index[:feature_num], scores, c_matrix

def PearsonRankAcc(classifier, X, y, feature_num):
    KofK_Fold = 5
    cv  = StratifiedKFold(n_splits=KofK_Fold)
    argsort_index, _ = pearson(X,y)
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    return classifier, argsort_index[:feature_num], scores, c_matrix

def McRankAcc(classifier, X, y, feature_num):
    KofK_Fold = 5
    cv  = StratifiedKFold(n_splits=KofK_Fold)
    argsort_index = MICoe(X,y)
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    return classifier, argsort_index[:feature_num], scores, c_matrix

def McOneAcc(classifier, X, y, feature_num):
    KofK_Fold = 5
    cv  = StratifiedKFold(n_splits=KofK_Fold)
    argsort_index = McOne(X,y)
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    return classifier, argsort_index[:feature_num], scores, c_matrix

def McTwoAcc(classifier, X, y, feature_num):
    if X.shape[0]>=20:
        nsplit = 10
    else:
        nsplit = 5
    cv  = StratifiedKFold(n_splits=nsplit)
    McTwoIndex = McTwo(X,y,feature_num)
    X_new = X[:,McTwoIndex]
    # if len(argsort_index)<feature_num:
    #     feature_num = len(argsort_index)
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    print("accuracy is : ",scores)
    return classifier, McTwoIndex, scores, c_matrix

"""
Selector-why
"""
def Ttest_RFE_SVM(features, label, fea_num, seed=None):
    p_value = []
    p_label = [k for k in range(len(label)) if label[k]==0]
    n_label = [k for k in range(len(label)) if label[k]!=0]
    for i in range(len(features[0])):
        t, p = stats.ttest_ind(
                features[p_label, i],
                features[n_label, i],
                equal_var=False)
        p_value.append(p)
    ttest_index = np.argsort(np.array(p_value))

    new_feature = features[:, np.array(ttest_index)]
    # new_feature = features[:, np.array(ttest_index)][:,:100]

    """RFE"""
    RFE_Linearsvc = LinearSVC(C=1, penalty="l1", dual=False, random_state=1)
    rfe = RFE(RFE_Linearsvc, n_features_to_select=fea_num)
    rfe.fit(new_feature, label)

    return np.array(ttest_index)[np.where(rfe.support_ == True)[0]],seed


def Ttest_RFE_SVM_loop(m,l,featureNum,preprocessingFlag=1,classifer=SVC(),KofK_Fold=10,showEachResult=0,RandomSeed=0):
    print(np.shape(m),np.shape(l))
    if(preprocessingFlag):
        m = preprocessing.scale(m)
    count=RandomSeed
    max_acc = 0
    max_matrix = 0
    tiiime = 0
    feanum_result = 0
    pre_label_max = None
    while(1):
        feanum,seed=Ttest_RFE_SVM(m, l, featureNum, count)
        
        X_new = m[:,np.array(feanum)]
#        if X_new.shape[1]==1:
#            X_new = X_new[:,np.array([0,0])]
#            X_new[:,1] = X_new[:,1]+np.sum(X_new[:,1])*0.01
        cv  = StratifiedKFold(n_splits=KofK_Fold)
        #scoring_fnc = make_scorer(balanced_accuracy_score)
        #scores = cross_val_score(classifer, X_new, l, cv=cv, scoring=scoring_fnc)

        c_matrix, pre_label = cross_val_confusion_matrix(classifer, X_new, l, cv)
        c_matrix_ = copy.deepcopy(c_matrix)
        c_matrix = np.sum(c_matrix,axis=0)
        scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2

        #scores = SKFold_accross_val_score_Grid(classifer, X_new, l)
        ######################################################
        if(showEachResult):
            print(scores)
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        if(scores.mean()>max_acc):
            tiiime=0
            seedMax=count
            max_acc=scores
            max_matrix = c_matrix_
            feanum_result = feanum
            pre_label_max = pre_label
            print(scores)
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        else:
            tiiime+=1
        count=count+10
        if tiiime>=0:
            break
    print(seedMax,max_acc)
    return classifer,feanum_result,max_acc,max_matrix,pre_label_max



=======
import numpy as np
import copy
import time
from collections import Counter
from scipy import stats
from minepy import MINE
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.model_selection import cross_validate,train_test_split,StratifiedKFold,cross_val_score,GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn import naive_bayes
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, fbeta_score, balanced_accuracy_score, confusion_matrix
from culAcc import cross_val_confusion_matrix
import warnings
warnings.filterwarnings("ignore")

"""
filter
"""
def ttest(X,y):
    p_value = []
    p_label = [k for k in range(len(y)) if y[k]==0]
    n_label = [k for k in range(len(y)) if y[k]!=0]
    for i in range(len(X[0])):
        t, p = stats.ttest_ind(
                X[p_label, i],
                X[n_label, i],
                equal_var=False)
        p_value.append(p)
    ttest_index = np.argsort(np.array(p_value))
    return ttest_index,np.array(p_value)

def wtest(X,y):
    y[y==-1]=0
    p_value = []
    p_label = [k for k in range(len(y)) if y[k]==0]
    n_label = [k for k in range(len(y)) if y[k]!=0]
    for i in range(len(X[0])):
        t, p = stats.mannwhitneyu(
                X[p_label, i],
                X[n_label, i],
                alternative="two-sided")
        p_value.append(p)
    ttest_index = np.argsort(np.array(p_value))
    return ttest_index,np.array(p_value)

def pearson(X, y):
    pearson_value = []
    for i in range(len(X[0])):
        r, p = stats.pearsonr(X[:, i], y)
        pearson_value.append(p)
    pearson_index = np.argsort(np.array(pearson_value))
    return pearson_index,np.array(pearson_value)

def spearman_test(X, y):
    spearman_value=[]
    for i in range(X.shape[1]):
        spearman_value.append(abs(stats.spearmanr(X[:,i],y)[0]))
    spearman_value_index = np.argsort(np.array(spearman_value))
    return spearman_value_index, np.array(spearman_value)

def MICoe(X, y, r=0.3, alpha=0.6, c=15):
    y[y==-1]=0
    mine = MINE(alpha=0.6, c=15)
    micFC = []
    Subset = []
    for i in range(X.shape[1]):
        mic = mine.compute_score(X[:,i],y)
        micFC.append(mine.mic())
    Subset = np.argsort(-np.array(micFC))
    return Subset

def McOne(X, y, r=0.2, alpha=0.6, c=15):
    y[y==-1]=0
    mine = MINE(alpha=0.6, c=15)
    micFC = []
    Subset = []
    Selected = []
    for i in range(X.shape[1]):
        mic = mine.compute_score(X[:,i],y)
        micFC.append(mine.mic())
    Subset = np.argsort(-np.array(micFC))
    e=0
    while(e<len(Subset[np.where(np.array(micFC)>=r)])):
        q = e+1
        while(q<len(Subset[Subset>=r])):
            mine.compute_score(X[:,Subset[q]],X[:,Subset[e]])
            if mine.mic()>=micFC[Subset[q]]:
                q+=1
            else:
                break
        Selected.append(Subset[e])
        e = q
    mine.compute_score(X[:,len(Subset[Subset>=r])-2],X[:,len(Subset[Subset>=r])-1])
    if mine.mic()<micFC[Subset[-1]]:
        Selected.append(Subset[-1])
    if len(Selected)>100:
        Selected = Selected[:100]
    return Selected

def McTwo(X,y,feanum):
    y[y==-1]=0
    McOneIndex = McOne(X,y)
    if X.shape[0]>=20:
        nsplit = 10
    else:
        nsplit = 5
    SelectedFeatureIndex = BFS(X[:,McOneIndex],y,"knn",fea_num=feanum,nsplit=nsplit)
    return np.array(McOneIndex)[SelectedFeatureIndex]

def BFS(X, y, clf, fea_num=30, nsplit=10, random_state=None):
    svmlabel = copy.deepcopy(y)
    svmlabel[svmlabel==0]=-1
    #base parameter
    if X.shape[0]<20:
        nsplit = 5
    scoring_fnc = make_scorer(balanced_accuracy_score)
    cv = StratifiedKFold(n_splits=nsplit, random_state=None)
    #clissifier
    #svm
    tuned_parameters_svc = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
                        #{'kernel': ['poly'] ,'degree':[10,11,12,13]}
                        ]
    clf_svc_grid = GridSearchCV(SVC(probability=True,random_state=None), tuned_parameters_svc,scoring=scoring_fnc)
    #knn
    tuned_parameters_knn = [
        # {
        #     'weights':['uniform'],
        #     'n_neighbors':[i for i in [3,5]]
        # },
        {
            'weights':['distance'],
            'n_neighbors':[i for i in [3,5]],
            #'p':[i for i in range(1,6)]
        }
    ]
    clf_knn_grid = GridSearchCV(KNeighborsClassifier(),tuned_parameters_knn,scoring=scoring_fnc)
    #bayes
    clf_bayes_grid = naive_bayes.GaussianNB()
    #Dtree
    tuned_parameters_dtree = {'max_depth':range(1,5),'criterion':np.array(['entropy','gini'])}
    clf_dtree_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), tuned_parameters_dtree,scoring=scoring_fnc)
    if clf=="svm":
        ccc = clf_svc_grid
    elif clf=="knn":
        ccc = clf_knn_grid
    elif clf=="bayes":
        ccc = clf_bayes_grid
    elif clf=="dtree":
        ccc = clf_dtree_grid
    else:
        raise AssertionError()
    ########################################################
    SelectedFeatureIndex = []
    FeatureIndex = [i for i in range(X.shape[1])]
    for i in range(min(fea_num,X.shape[1])):
        max_index = None
        max_acc = 0
        for j in FeatureIndex:
            # t1 = time.time()
            acc = np.mean(cross_val_score(ccc,X[:,SelectedFeatureIndex+[j]],y))
            # print("time is : ",time.time()-t1)
            # print(acc)
            if acc>max_acc:
                max_acc = acc
                max_index = j
        # try:
        if max_index==None:
            print("pause")
        SelectedFeatureIndex.append(max_index)
        FeatureIndex.remove(max_index)
        # except:
        #     print(X.shape)
        #     print(max_index)
        #     raise AssertionError()
    return SelectedFeatureIndex


"""
feature-select and cul acc
"""

def TRankAcc(classifier, X, y, feature_num):
    if X.shape[0]<20:
        nsplit = 5
    else:
        nsplit = 10
    cv  = StratifiedKFold(n_splits=nsplit)
    argsort_index, _ = ttest(X,y)
    TrankIndex = BFS(X[:,argsort_index], y ,"knn", fea_num=feature_num, nsplit=nsplit)
    argsort_index = np.array(argsort_index)[TrankIndex]
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    print("accuracy is : ",scores)
    return classifier, argsort_index[:feature_num], scores, c_matrix

def WRankAcc(classifier, X, y, feature_num):
    if X.shape[0]<20:
        nsplit = 5
    else:
        nsplit = 10
    cv  = StratifiedKFold(n_splits=nsplit)
    argsort_index, _ = wtest(X,y)
    TrankIndex = BFS(X[:,argsort_index], y ,"knn", fea_num=feature_num, nsplit=nsplit)
    argsort_index = np.array(argsort_index)[TrankIndex]
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    print("accuracy is : ",scores)
    return classifier, argsort_index[:feature_num], scores, c_matrix

def PearsonRankAcc(classifier, X, y, feature_num):
    KofK_Fold = 5
    cv  = StratifiedKFold(n_splits=KofK_Fold)
    argsort_index, _ = pearson(X,y)
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    return classifier, argsort_index[:feature_num], scores, c_matrix

def McRankAcc(classifier, X, y, feature_num):
    KofK_Fold = 5
    cv  = StratifiedKFold(n_splits=KofK_Fold)
    argsort_index = MICoe(X,y)
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    return classifier, argsort_index[:feature_num], scores, c_matrix

def McOneAcc(classifier, X, y, feature_num):
    KofK_Fold = 5
    cv  = StratifiedKFold(n_splits=KofK_Fold)
    argsort_index = McOne(X,y)
    X_new = X[:,argsort_index[:feature_num]]
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    return classifier, argsort_index[:feature_num], scores, c_matrix

def McTwoAcc(classifier, X, y, feature_num):
    if X.shape[0]>=20:
        nsplit = 10
    else:
        nsplit = 5
    cv  = StratifiedKFold(n_splits=nsplit)
    McTwoIndex = McTwo(X,y,feature_num)
    X_new = X[:,McTwoIndex]
    # if len(argsort_index)<feature_num:
    #     feature_num = len(argsort_index)
    l = y
    c_matrix, _ = cross_val_confusion_matrix(classifier, X_new, l, cv)
    c_matrix = np.sum(c_matrix,axis=0)
    scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2
    print("accuracy is : ",scores)
    return classifier, McTwoIndex, scores, c_matrix

"""
Selector-why
"""
def Ttest_RFE_SVM(features, label, fea_num, seed=None):
    p_value = []
    p_label = [k for k in range(len(label)) if label[k]==0]
    n_label = [k for k in range(len(label)) if label[k]!=0]
    for i in range(len(features[0])):
        t, p = stats.ttest_ind(
                features[p_label, i],
                features[n_label, i],
                equal_var=False)
        p_value.append(p)
    ttest_index = np.argsort(np.array(p_value))

    new_feature = features[:, np.array(ttest_index)]
    # new_feature = features[:, np.array(ttest_index)][:,:100]

    """RFE"""
    RFE_Linearsvc = LinearSVC(C=1, penalty="l1", dual=False, random_state=1)
    rfe = RFE(RFE_Linearsvc, n_features_to_select=fea_num)
    rfe.fit(new_feature, label)

    return np.array(ttest_index)[np.where(rfe.support_ == True)[0]],seed


def Ttest_RFE_SVM_loop(m,l,featureNum,preprocessingFlag=1,classifer=SVC(),KofK_Fold=10,showEachResult=0,RandomSeed=0):
    print(np.shape(m),np.shape(l))
    if(preprocessingFlag):
        m = preprocessing.scale(m)
    count=RandomSeed
    max_acc = 0
    max_matrix = 0
    tiiime = 0
    feanum_result = 0
    pre_label_max = None
    while(1):
        feanum,seed=Ttest_RFE_SVM(m, l, featureNum, count)
        
        X_new = m[:,np.array(feanum)]
#        if X_new.shape[1]==1:
#            X_new = X_new[:,np.array([0,0])]
#            X_new[:,1] = X_new[:,1]+np.sum(X_new[:,1])*0.01
        cv  = StratifiedKFold(n_splits=KofK_Fold)
        #scoring_fnc = make_scorer(balanced_accuracy_score)
        #scores = cross_val_score(classifer, X_new, l, cv=cv, scoring=scoring_fnc)

        c_matrix, pre_label = cross_val_confusion_matrix(classifer, X_new, l, cv)
        c_matrix_ = copy.deepcopy(c_matrix)
        c_matrix = np.sum(c_matrix,axis=0)
        scores = ((c_matrix[0]/(c_matrix[0]+c_matrix[1]))+(c_matrix[3]/(c_matrix[2]+c_matrix[3])))/2

        #scores = SKFold_accross_val_score_Grid(classifer, X_new, l)
        ######################################################
        if(showEachResult):
            print(scores)
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        if(scores.mean()>max_acc):
            tiiime=0
            seedMax=count
            max_acc=scores
            max_matrix = c_matrix_
            feanum_result = feanum
            pre_label_max = pre_label
            print(scores)
            print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
        else:
            tiiime+=1
        count=count+10
        if tiiime>=0:
            break
    print(seedMax,max_acc)
    return classifer,feanum_result,max_acc,max_matrix,pre_label_max



>>>>>>> c2a53a595f4c57c357d58c4c764e7338c4532a05
