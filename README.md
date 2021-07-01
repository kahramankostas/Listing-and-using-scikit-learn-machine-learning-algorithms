
# Listing and using scikit-learn machine learning algorithms

#### This file lists the machine learning algorithms included in scikit-learn and classifies the Wisconsin breast cancer dataset using these algorithms. 5 of these algorithms was optimized using the RandomizedSearchCV method.  The optimized algorithms are:

* RandomForestClassifier
* ExtraTreeClassifier
* SVC
* GradientBoostingClassifier
* DecisionTreeClassifier

### importing of required libraries for listing Sk-learn estimators


```
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings("ignore")


```


```
estimators = all_estimators(type_filter='classifier')

estimator_list = []
for number, estimator in estimators:
    print(number, estimator)
    try:
        clf = estimator()
        estimator_list.append(clf)
    except Exception as e:
        print('\nUnable to import   ------------------>', estimator,"\n")
        print(e)

```

    AdaBoostClassifier <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>
    BaggingClassifier <class 'sklearn.ensemble._bagging.BaggingClassifier'>
    BernoulliNB <class 'sklearn.naive_bayes.BernoulliNB'>
    CalibratedClassifierCV <class 'sklearn.calibration.CalibratedClassifierCV'>
    CategoricalNB <class 'sklearn.naive_bayes.CategoricalNB'>
    ClassifierChain <class 'sklearn.multioutput.ClassifierChain'>
    
    Unable to import   ------------------> <class 'sklearn.multioutput.ClassifierChain'> 
    
    __init__() missing 1 required positional argument: 'base_estimator'
    ComplementNB <class 'sklearn.naive_bayes.ComplementNB'>
    DecisionTreeClassifier <class 'sklearn.tree._classes.DecisionTreeClassifier'>
    DummyClassifier <class 'sklearn.dummy.DummyClassifier'>
    ExtraTreeClassifier <class 'sklearn.tree._classes.ExtraTreeClassifier'>
    ExtraTreesClassifier <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>
    GaussianNB <class 'sklearn.naive_bayes.GaussianNB'>
    GaussianProcessClassifier <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>
    GradientBoostingClassifier <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>
    HistGradientBoostingClassifier <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>
    KNeighborsClassifier <class 'sklearn.neighbors._classification.KNeighborsClassifier'>
    LabelPropagation <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>
    LabelSpreading <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>
    LinearDiscriminantAnalysis <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>
    LinearSVC <class 'sklearn.svm._classes.LinearSVC'>
    LogisticRegression <class 'sklearn.linear_model._logistic.LogisticRegression'>
    LogisticRegressionCV <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>
    MLPClassifier <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>
    MultiOutputClassifier <class 'sklearn.multioutput.MultiOutputClassifier'>
    
    Unable to import   ------------------> <class 'sklearn.multioutput.MultiOutputClassifier'> 
    
    __init__() missing 1 required positional argument: 'estimator'
    MultinomialNB <class 'sklearn.naive_bayes.MultinomialNB'>
    NearestCentroid <class 'sklearn.neighbors._nearest_centroid.NearestCentroid'>
    NuSVC <class 'sklearn.svm._classes.NuSVC'>
    OneVsOneClassifier <class 'sklearn.multiclass.OneVsOneClassifier'>
    
    Unable to import   ------------------> <class 'sklearn.multiclass.OneVsOneClassifier'> 
    
    __init__() missing 1 required positional argument: 'estimator'
    OneVsRestClassifier <class 'sklearn.multiclass.OneVsRestClassifier'>
    
    Unable to import   ------------------> <class 'sklearn.multiclass.OneVsRestClassifier'> 
    
    __init__() missing 1 required positional argument: 'estimator'
    OutputCodeClassifier <class 'sklearn.multiclass.OutputCodeClassifier'>
    
    Unable to import   ------------------> <class 'sklearn.multiclass.OutputCodeClassifier'> 
    
    __init__() missing 1 required positional argument: 'estimator'
    PassiveAggressiveClassifier <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>
    Perceptron <class 'sklearn.linear_model._perceptron.Perceptron'>
    QuadraticDiscriminantAnalysis <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>
    RadiusNeighborsClassifier <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>
    RandomForestClassifier <class 'sklearn.ensemble._forest.RandomForestClassifier'>
    RidgeClassifier <class 'sklearn.linear_model._ridge.RidgeClassifier'>
    RidgeClassifierCV <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>
    SGDClassifier <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>
    SVC <class 'sklearn.svm._classes.SVC'>
    StackingClassifier <class 'sklearn.ensemble._stacking.StackingClassifier'>
    
    Unable to import   ------------------> <class 'sklearn.ensemble._stacking.StackingClassifier'> 
    
    __init__() missing 1 required positional argument: 'estimators'
    VotingClassifier <class 'sklearn.ensemble._voting.VotingClassifier'>
    
    Unable to import   ------------------> <class 'sklearn.ensemble._voting.VotingClassifier'> 
    
    __init__() missing 1 required positional argument: 'estimators'
    

### list of obtained algorithms


```
estimator_list
```




    [AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                        n_estimators=50, random_state=None),
     BaggingClassifier(base_estimator=None, bootstrap=True, bootstrap_features=False,
                       max_features=1.0, max_samples=1.0, n_estimators=10,
                       n_jobs=None, oob_score=False, random_state=None, verbose=0,
                       warm_start=False),
     BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True),
     CalibratedClassifierCV(base_estimator=None, cv=None, method='sigmoid'),
     CategoricalNB(alpha=1.0, class_prior=None, fit_prior=True),
     ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False),
     DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                            max_depth=None, max_features=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, presort='deprecated',
                            random_state=None, splitter='best'),
     DummyClassifier(constant=None, random_state=None, strategy='warn'),
     ExtraTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, random_state=None,
                         splitter='random'),
     ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                          criterion='gini', max_depth=None, max_features='auto',
                          max_leaf_nodes=None, max_samples=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=None, verbose=0,
                          warm_start=False),
     GaussianNB(priors=None, var_smoothing=1e-09),
     GaussianProcessClassifier(copy_X_train=True, kernel=None, max_iter_predict=100,
                               multi_class='one_vs_rest', n_jobs=None,
                               n_restarts_optimizer=0, optimizer='fmin_l_bfgs_b',
                               random_state=None, warm_start=False),
     GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                learning_rate=0.1, loss='deviance', max_depth=3,
                                max_features=None, max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=100,
                                n_iter_no_change=None, presort='deprecated',
                                random_state=None, subsample=1.0, tol=0.0001,
                                validation_fraction=0.1, verbose=0,
                                warm_start=False),
     HistGradientBoostingClassifier(l2_regularization=0.0, learning_rate=0.1,
                                    loss='auto', max_bins=255, max_depth=None,
                                    max_iter=100, max_leaf_nodes=31,
                                    min_samples_leaf=20, n_iter_no_change=None,
                                    random_state=None, scoring=None, tol=1e-07,
                                    validation_fraction=0.1, verbose=0,
                                    warm_start=False),
     KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                          metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                          weights='uniform'),
     LabelPropagation(gamma=20, kernel='rbf', max_iter=1000, n_jobs=None,
                      n_neighbors=7, tol=0.001),
     LabelSpreading(alpha=0.2, gamma=20, kernel='rbf', max_iter=30, n_jobs=None,
                    n_neighbors=7, tol=0.001),
     LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                solver='svd', store_covariance=False, tol=0.0001),
     LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
               intercept_scaling=1, loss='squared_hinge', max_iter=1000,
               multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
               verbose=0),
     LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None, max_iter=100,
                        multi_class='auto', n_jobs=None, penalty='l2',
                        random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                        warm_start=False),
     LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                          fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                          max_iter=100, multi_class='auto', n_jobs=None,
                          penalty='l2', random_state=None, refit=True, scoring=None,
                          solver='lbfgs', tol=0.0001, verbose=0),
     MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                   beta_2=0.999, early_stopping=False, epsilon=1e-08,
                   hidden_layer_sizes=(100,), learning_rate='constant',
                   learning_rate_init=0.001, max_fun=15000, max_iter=200,
                   momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
                   power_t=0.5, random_state=None, shuffle=True, solver='adam',
                   tol=0.0001, validation_fraction=0.1, verbose=False,
                   warm_start=False),
     MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True),
     NearestCentroid(metric='euclidean', shrink_threshold=None),
     NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
           decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
           max_iter=-1, nu=0.5, probability=False, random_state=None, shrinking=True,
           tol=0.001, verbose=False),
     PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                                 early_stopping=False, fit_intercept=True,
                                 loss='hinge', max_iter=1000, n_iter_no_change=5,
                                 n_jobs=None, random_state=None, shuffle=True,
                                 tol=0.001, validation_fraction=0.1, verbose=0,
                                 warm_start=False),
     Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
                fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
                penalty=None, random_state=0, shuffle=True, tol=0.001,
                validation_fraction=0.1, verbose=0, warm_start=False),
     QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                                   store_covariance=False, tol=0.0001),
     RadiusNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                               metric_params=None, n_jobs=None, outlier_label=None,
                               p=2, radius=1.0, weights='uniform'),
     RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=None, max_features='auto',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=100,
                            n_jobs=None, oob_score=False, random_state=None,
                            verbose=0, warm_start=False),
     RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                     max_iter=None, normalize=False, random_state=None,
                     solver='auto', tol=0.001),
     RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ]), class_weight=None, cv=None,
                       fit_intercept=True, normalize=False, scoring=None,
                       store_cv_values=False),
     SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                   early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                   l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                   max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='l2',
                   power_t=0.5, random_state=None, shuffle=True, tol=0.001,
                   validation_fraction=0.1, verbose=0, warm_start=False),
     SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
         decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
         max_iter=-1, probability=False, random_state=None, shrinking=True,
         tol=0.001, verbose=False)]



### importing required libraries for using this MLs


```
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import  ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier


from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

from sklearn import datasets
import sklearn
import time


from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import randint as sp_randFloat
```

### Load the breast cancer wisconsin dataset


```
cancer=datasets.load_breast_cancer()


X_cancer = cancer.data
y_cancer= cancer.target
```

###  List of MLs in dictinary


```
ml_list={"ExtraTreeClassifier":ExtraTreeClassifier(),
"DecisionTreeClassifier":DecisionTreeClassifier(),
"OneClassSVM":OneClassSVM(),
"MLPClassifier":MLPClassifier(),
"ComplementNB":ComplementNB(),
"DummyClassifier":DummyClassifier(),         
"RadiusNeighborsClassifier":RadiusNeighborsClassifier(),
"KNeighborsClassifier":KNeighborsClassifier(),
"ClassifierChain":ClassifierChain(base_estimator=DecisionTreeClassifier()),
"MultiOutputClassifier":MultiOutputClassifier(estimator=DecisionTreeClassifier()),
"OutputCodeClassifier":OutputCodeClassifier(estimator=DecisionTreeClassifier()),
"OneVsOneClassifier":OneVsOneClassifier(estimator=DecisionTreeClassifier()),
"OneVsRestClassifier":OneVsRestClassifier(estimator=DecisionTreeClassifier()),
"SGDClassifier":SGDClassifier(),
"RidgeClassifierCV":RidgeClassifierCV(),
"RidgeClassifier":RidgeClassifier(),
"PassiveAggressiveClassifier    ":PassiveAggressiveClassifier    (),
"GaussianProcessClassifier":GaussianProcessClassifier(),
"AdaBoostClassifier":AdaBoostClassifier(),
"GradientBoostingClassifier":GradientBoostingClassifier(),
"BaggingClassifier":BaggingClassifier(),
"ExtraTreesClassifier":ExtraTreesClassifier(),
"RandomForestClassifier":RandomForestClassifier(),
"BernoulliNB":BernoulliNB(),
"CalibratedClassifierCV":CalibratedClassifierCV(),
"GaussianNB":GaussianNB(),
"LabelPropagation":LabelPropagation(),
"LabelSpreading":LabelSpreading(),
"LinearDiscriminantAnalysis":LinearDiscriminantAnalysis(),
"LinearSVC":LinearSVC(),
"LogisticRegression":LogisticRegression(),
"LogisticRegressionCV":LogisticRegressionCV(),
"MultinomialNB  ":MultinomialNB  (),
"NearestCentroid":NearestCentroid(),
"NuSVC":NuSVC(),
"Perceptron":Perceptron(),
"QuadraticDiscriminantAnalysis":QuadraticDiscriminantAnalysis(),
"SVC":SVC(),
"HistGradientBoostingClassifier":HistGradientBoostingClassifier(),
"CategoricalNB" : CategoricalNB()}

```


```
X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size = 0.25, random_state = 0)

print ('%-40s %-20s %-20s %-20s %-20s' % ("Model".center(22) ,"F1 Score".center(20),"Accuracy".center(15) ,"Training Time".center(15),"Testing Time".center(15) ))
print ('%-40s %-20s %-20s %-20s %-20s' % ("|____________________|","____________________","____________________" ,"____________________","____________________" ))
#X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size = 0.25, random_state = 0)

for i in ml_list:
    try:
        clf=ml_list[i]
        second=time.time()
        clf.fit(X_train, y_train)
        train=round(time.time()-second,5)
        second=time.time()
        predict =clf.predict(X_test)
        test=round(time.time()-second,5)
        f1=round(sklearn.metrics.f1_score(y_test, predict, average='macro'),5)
        acc=round(sklearn.metrics.accuracy_score(y_test, predict),5)
        print ('%-40s %-20s %-20s %-20s %-20s' % (i,f1,acc,train,test ))
    except:        print ('%-40s %-20s %-20s' % (i,"Error","Error" ))

```

            Model                                  F1 Score           Accuracy          Training Time         Testing Time      
    |____________________|                   ____________________ ____________________ ____________________ ____________________
    ExtraTreeClassifier                      0.92614              0.93007              0.001                0.0                 
    DecisionTreeClassifier                   0.89092              0.8951               0.00698              0.0                 
    OneClassSVM                              0.20513              0.33566              0.012                0.00196             
    MLPClassifier                            0.89586              0.9021               0.46438              0.001               
    ComplementNB                             0.89024              0.9021               0.001                0.00101             
    DummyClassifier                          0.55368              0.58042              0.001                0.0                 
    RadiusNeighborsClassifier                Error                Error               
    KNeighborsClassifier                     0.9328               0.93706              0.001                0.00698             
    ClassifierChain                          Error                Error               
    MultiOutputClassifier                    Error                Error               
    OutputCodeClassifier                     0.27041              0.37063              0.001                0.00101             
    OneVsOneClassifier                       0.88398              0.88811              0.00598              0.0                 
    OneVsRestClassifier                      0.90491              0.90909              0.00598              0.001               
    SGDClassifier                            0.93905              0.94406              0.00199              0.0                 
    RidgeClassifierCV                        0.95387              0.95804              0.00399              0.0                 
    RidgeClassifier                          0.95387              0.95804              0.00299              0.0                 
    PassiveAggressiveClassifier              0.75481              0.75524              0.001                0.001               
    GaussianProcessClassifier                0.91787              0.92308              0.08777              0.00997             
    AdaBoostClassifier                       0.9776               0.97902              0.13169              0.00797             
    GradientBoostingClassifier               0.97025              0.97203              0.39192              0.00103             
    BaggingClassifier                        0.94812              0.95105              0.05186              0.00203             
    ExtraTreesClassifier                     0.97002              0.97203              0.11665              0.01795             
    RandomForestClassifier                   0.96267              0.96503              0.20944              0.01396             
    BernoulliNB                              0.38627              0.62937              0.00199              0.0                 
    CalibratedClassifierCV                   0.93849              0.94406              0.11672              0.001               
    GaussianNB                               0.93228              0.93706              0.00096              0.001               
    LabelPropagation                         0.33012              0.40559              0.00994              0.00299             
    LabelSpreading                           0.33012              0.40559              0.01197              0.00499             
    LinearDiscriminantAnalysis               0.96952              0.97203              0.00499              0.0                 
    LinearSVC                                0.70593              0.70629              0.02597              0.0                 
    LogisticRegression                       0.95568              0.95804              0.0628               0.0                 
    LogisticRegressionCV                     0.96294              0.96503              2.10138              0.0                 
    MultinomialNB                            0.89024              0.9021               0.0                  0.0                 
    NearestCentroid                          0.88907              0.9021               0.0                  0.0                 
    NuSVC                                    0.863                0.88112              0.00701              0.00199             
    Perceptron                               0.82423              0.85315              0.00196              0.0                 
    QuadraticDiscriminantAnalysis            0.95537              0.95804              0.001                0.0                 
    SVC                                      0.93048              0.93706              0.00299              0.001               
    HistGradientBoostingClassifier           0.97743              0.97902              0.93048              0.00399             
    CategoricalNB                            Error                Error               
    

# optimizing some MLs using the RandomizedSearchCV method


```
opt = {"RandomForestClassifier":{"max_depth":np.linspace(1, 32, 32, endpoint=True),
"n_estimators" : sp_randint(1, 200),
"max_features": sp_randint(1, 11),
"min_samples_split":sp_randint(2, 11),
"bootstrap": [True, False],
"criterion": ["gini", "entropy"]},
          
"ExtraTreeClassifier":{"max_depth":np.linspace(1, 32, 32, endpoint=True),
"max_features": sp_randint(1, 11),
"min_samples_split":sp_randint(2, 11),
#"ccp_alpha":sp_randint(2, 11),
#"class_weight":["balanced", "balanced_subsample"],"max_leaf_nodes"
"criterion": ["gini", "entropy"]},

"SVC": {"C": np.linspace(1, 1000, 10000, endpoint=True),
"gamma": np.linspace(0.1, 1000, 10000, endpoint=True)},
          
"GradientBoostingClassifier":{"learning_rate":np.linspace(0.0, 100, 10000, endpoint=True), #sp_randFloat(0.2,1.0),
"subsample"    :np.linspace(0.0, 1, 100, endpoint=True), #sp_randFloat(0.2,1.0),
"n_estimators" : sp_randInt(1, 1000),
"max_depth"    : sp_randInt(1, 1000)},       

          
          
"DecisionTreeClassifier" :  { 'criterion':['gini','entropy'],
"max_depth":np.linspace(1, 100, 100, endpoint=True),
"min_samples_split": sp_randint(2,100),#uniform(0.1,1),
 #"min_samples_leafs" : np.linspace(0.1, 0.5, 5, endpoint=True),
"max_features" : sp_randint(1,X_train.shape[1])}    
         }
```


```
X = cancer.data
y= cancer.target
models=[RandomForestClassifier(),
ExtraTreeClassifier(),
SVC(),
GradientBoostingClassifier(),
DecisionTreeClassifier()]
for i in models:
    
    clf = i#(n_estimators=20)
    second=time.time()
    # use a full grid over all parameters
    temp=str(i)[:-2]
    print(temp)
    for ii in [0,1]:
        if ii:
            param_dist =   opt[temp]
            print("OPTIMIZED")
        else:
            param_dist={}
            print("NOT OPTIMIZED")
        n_iter_search = 10
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist)
        random_search.fit(X, y)
        print(random_search.best_params_)
        print (random_search.best_score_)
        print (random_search.best_params_)
        print (random_search.best_estimator_)
        print("time= ", (time.time()-second))
        print("-------------------------------------------------------------")
```

    RandomForestClassifier
    NOT OPTIMIZED
    {}
    0.9578481602235678
    {}
    RandomForestClassifier()
    time=  1.186774730682373
    -------------------------------------------------------------
    OPTIMIZED
    {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 10.0, 'max_features': 6, 'min_samples_split': 5, 'n_estimators': 155}
    0.9701443875174661
    {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 10.0, 'max_features': 6, 'min_samples_split': 5, 'n_estimators': 155}
    RandomForestClassifier(bootstrap=False, criterion='entropy', max_depth=10.0,
                           max_features=6, min_samples_split=5, n_estimators=155)
    time=  11.857276439666748
    -------------------------------------------------------------
    ExtraTreeClassifier
    NOT OPTIMIZED
    {}
    0.924468250271697
    {}
    ExtraTreeClassifier()
    time=  0.012934684753417969
    -------------------------------------------------------------
    OPTIMIZED
    {'criterion': 'gini', 'max_depth': 10.0, 'max_features': 8, 'min_samples_split': 4}
    0.947259742276044
    {'criterion': 'gini', 'max_depth': 10.0, 'max_features': 8, 'min_samples_split': 4}
    ExtraTreeClassifier(max_depth=10.0, max_features=8, min_samples_split=4)
    time=  0.08474206924438477
    -------------------------------------------------------------
    SVC
    NOT OPTIMIZED
    {}
    0.9121720229777983
    {}
    SVC()
    time=  0.03490734100341797
    -------------------------------------------------------------
    OPTIMIZED
    {'gamma': 572.6, 'C': 63.94329432943294}
    0.6274181027790716
    {'gamma': 572.6, 'C': 63.94329432943294}
    SVC(C=63.94329432943294, gamma=572.6)
    time=  1.5508544445037842
    -------------------------------------------------------------
    GradientBoostingClassifier
    NOT OPTIMIZED
    {}
    0.9578636857630801
    {}
    GradientBoostingClassifier()
    time=  2.1522772312164307
    -------------------------------------------------------------
    OPTIMIZED
    {'learning_rate': 87.52875287528754, 'max_depth': 560, 'n_estimators': 853, 'subsample': 0.9393939393939394}
    0.9209594783418724
    {'learning_rate': 87.52875287528754, 'max_depth': 560, 'n_estimators': 853, 'subsample': 0.9393939393939394}
    GradientBoostingClassifier(learning_rate=87.52875287528754, max_depth=560,
                               n_estimators=853, subsample=0.9393939393939394)
    time=  83.62071204185486
    -------------------------------------------------------------
    DecisionTreeClassifier
    NOT OPTIMIZED
    {}
    0.915618692749573
    {}
    DecisionTreeClassifier()
    time=  0.03889608383178711
    -------------------------------------------------------------
    OPTIMIZED
    {'criterion': 'entropy', 'max_depth': 36.0, 'max_features': 28, 'min_samples_split': 15}
    0.9420121099208197
    {'criterion': 'entropy', 'max_depth': 36.0, 'max_features': 28, 'min_samples_split': 15}
    DecisionTreeClassifier(criterion='entropy', max_depth=36.0, max_features=28,
                           min_samples_split=15)
    time=  0.2752647399902344
    -------------------------------------------------------------
    
