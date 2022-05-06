# Listing and using scikit-learn machine learning algorithms

#### This file lists the machine learning algorithms included in scikit-learn and classifies the Wisconsin breast cancer dataset using these algorithms. 5 of these algorithms was optimized using the RandomizedSearchCV method.  The optimized algorithms are:

* RandomForestClassifier
* ExtraTreeClassifier
* SVC
* GradientBoostingClassifier
* DecisionTreeClassifier

### importing of required libraries for listing Sk-learn estimators


```python
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings("ignore")


```


```python
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


```python
estimator_list
```




    [AdaBoostClassifier(),
     BaggingClassifier(),
     BernoulliNB(),
     CalibratedClassifierCV(),
     CategoricalNB(),
     ComplementNB(),
     DecisionTreeClassifier(),
     DummyClassifier(),
     ExtraTreeClassifier(),
     ExtraTreesClassifier(),
     GaussianNB(),
     GaussianProcessClassifier(),
     GradientBoostingClassifier(),
     HistGradientBoostingClassifier(),
     KNeighborsClassifier(),
     LabelPropagation(),
     LabelSpreading(),
     LinearDiscriminantAnalysis(),
     LinearSVC(),
     LogisticRegression(),
     LogisticRegressionCV(),
     MLPClassifier(),
     MultinomialNB(),
     NearestCentroid(),
     NuSVC(),
     PassiveAggressiveClassifier(),
     Perceptron(),
     QuadraticDiscriminantAnalysis(),
     RadiusNeighborsClassifier(),
     RandomForestClassifier(),
     RidgeClassifier(),
     RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ])),
     SGDClassifier(),
     SVC()]



### importing required libraries for using this MLs


```python
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import OneClassSVM
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier    
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
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


```python
cancer=datasets.load_breast_cancer()


X_cancer = cancer.data
y_cancer= cancer.target
```

###  List of MLs in dictinary


```python
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


```python
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
    ExtraTreeClassifier                      0.95467              0.95804              0.00756              0.00199             
    DecisionTreeClassifier                   0.89206              0.8951               0.01278              0.00071             
    OneClassSVM                              0.20513              0.33566              0.02194              0.01895             
    MLPClassifier                            0.93228              0.93706              12.49243             0.01508             
    ComplementNB                             0.89024              0.9021               0.002                0.0                 
    DummyClassifier                          0.38627              0.62937              0.00087              0.00012             
    RadiusNeighborsClassifier                Error                Error               
    KNeighborsClassifier                     0.9328               0.93706              0.0011               0.0378              
    ClassifierChain                          Error                Error               
    MultiOutputClassifier                    Error                Error               
    OutputCodeClassifier                     0.85643              0.86014              0.01596              0.001               
    OneVsOneClassifier                       0.88331              0.88811              0.01283              0.001               
    OneVsRestClassifier                      0.89206              0.8951               0.02094              0.01795             
    SGDClassifier                            0.60135              0.60839              0.00808              0.0                 
    RidgeClassifierCV                        0.95387              0.95804              0.02493              0.0                 
    RidgeClassifier                          0.95387              0.95804              0.00997              0.0                 
    PassiveAggressiveClassifier              0.66234              0.66434              0.00388              0.0                 
    GaussianProcessClassifier                0.91787              0.92308              1.34211              0.02886             
    AdaBoostClassifier                       0.9776               0.97902              0.39993              0.03105             
    GradientBoostingClassifier               0.97025              0.97203              1.13681              0.00108             
    BaggingClassifier                        0.95568              0.95804              0.12886              0.00299             
    ExtraTreesClassifier                     0.95503              0.95804              0.33228              0.03383             
    RandomForestClassifier                   0.9776               0.97902              0.57395              0.03713             
    BernoulliNB                              0.38627              0.62937              0.00271              0.00028             
    CalibratedClassifierCV                   0.93849              0.94406              0.38996              0.00499             
    GaussianNB                               0.93228              0.93706              0.00199              0.00106             
    LabelPropagation                         0.33012              0.40559              0.04623              0.01895             
    LabelSpreading                           0.33012              0.40559              0.05785              0.03191             
    LinearDiscriminantAnalysis               0.96952              0.97203              0.01795              0.0                 
    LinearSVC                                0.94689              0.95105              0.06834              0.0                 
    LogisticRegression                       0.95568              0.95804              0.10372              0.001               
    LogisticRegressionCV                     0.96294              0.96503              4.77258              0.001               
    MultinomialNB                            0.89024              0.9021               0.00199              0.0                 
    NearestCentroid                          0.88907              0.9021               0.00126              0.00073             
    NuSVC                                    0.863                0.88112              0.02394              0.01795             
    Perceptron                               0.82423              0.85315              0.0031               0.0                 
    QuadraticDiscriminantAnalysis            0.95537              0.95804              0.00997              0.00013             
    SVC                                      0.93048              0.93706              0.00898              0.00598             
    HistGradientBoostingClassifier           0.97743              0.97902              153.78106            2.00046             
    CategoricalNB                            Error                Error               
    

# optimizing some MLs using the RandomizedSearchCV method


```python
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


```python
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
    0.9613724576929048
    {}
    RandomForestClassifier()
    time=  3.730778217315674
    -------------------------------------------------------------
    OPTIMIZED
    {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 16.0, 'max_features': 6, 'min_samples_split': 7, 'n_estimators': 50}
    0.9683900015525537
    {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 16.0, 'max_features': 6, 'min_samples_split': 7, 'n_estimators': 50}
    RandomForestClassifier(criterion='entropy', max_depth=16.0, max_features=6,
                           min_samples_split=7, n_estimators=50)
    time=  30.43511438369751
    -------------------------------------------------------------
    ExtraTreeClassifier
    NOT OPTIMIZED
    {}
    0.9173886042539978
    {}
    ExtraTreeClassifier()
    time=  0.022939205169677734
    -------------------------------------------------------------
    OPTIMIZED
    {'criterion': 'entropy', 'max_depth': 29.0, 'max_features': 8, 'min_samples_split': 9}
    0.9349324639031208
    {'criterion': 'entropy', 'max_depth': 29.0, 'max_features': 8, 'min_samples_split': 9}
    ExtraTreeClassifier(criterion='entropy', max_depth=29.0, max_features=8,
                        min_samples_split=9)
    time=  0.29825901985168457
    -------------------------------------------------------------
    SVC
    NOT OPTIMIZED
    {}
    0.9121720229777983
    {}
    SVC()
    time=  0.13563799858093262
    -------------------------------------------------------------
    OPTIMIZED
    {'gamma': 290.3, 'C': 529.024302430243}
    0.6274181027790716
    {'gamma': 290.3, 'C': 529.024302430243}
    SVC(C=529.024302430243, gamma=290.3)
    time=  6.7241644859313965
    -------------------------------------------------------------
    GradientBoostingClassifier
    NOT OPTIMIZED
    {}
    0.9596180717279925
    {}
    GradientBoostingClassifier()
    time=  7.962490558624268
    -------------------------------------------------------------
    OPTIMIZED
    {'learning_rate': 2.8402840284028406, 'max_depth': 207, 'n_estimators': 707, 'subsample': 0.38383838383838387}
    0.9174196553330228
    {'learning_rate': 2.8402840284028406, 'max_depth': 207, 'n_estimators': 707, 'subsample': 0.38383838383838387}
    GradientBoostingClassifier(learning_rate=2.8402840284028406, max_depth=207,
                               n_estimators=707, subsample=0.38383838383838387)
    time=  166.2842981815338
    -------------------------------------------------------------
    DecisionTreeClassifier
    NOT OPTIMIZED
    {}
    0.9191274646793974
    {}
    DecisionTreeClassifier()
    time=  0.12399601936340332
    -------------------------------------------------------------
    OPTIMIZED
    {'criterion': 'entropy', 'max_depth': 6.0, 'max_features': 6, 'min_samples_split': 23}
    0.9297469337059463
    {'criterion': 'entropy', 'max_depth': 6.0, 'max_features': 6, 'min_samples_split': 23}
    DecisionTreeClassifier(criterion='entropy', max_depth=6.0, max_features=6,
                           min_samples_split=23)
    time=  0.8205883502960205
    -------------------------------------------------------------
    


```python

```
