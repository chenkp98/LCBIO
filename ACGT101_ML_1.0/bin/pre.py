import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from xgboost import XGBClassifier as XGB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingClassifier as GBDT, AdaBoostClassifier as ADA
from sklearn.naive_bayes import GaussianNB as GNB


def pre_progress(x):                
    from sklearn.preprocessing import MinMaxScaler              
    scaler = MinMaxScaler().fit_transform(x)  
    X = pd.DataFrame(scaler,columns=x.columns)           
    return X                


def preprogress(X, y):
    classifiers = {             
            "逻辑回归": LR(),                             
            "支持向量机": SVC(),                
            "k近邻": KNN(),             
            "决策树": DT(),             
            "随机森林": RF(),               
            "XGBoost": XGB(),               
            "线性判别分析": LDA(),              
            "GBDT": GBDT(),             
            "ADABoost": ADA(),              
            "朴素贝叶斯": GNB()             
        }               
    results = {}                   
    for classifier_name, classifier in classifiers.items():             
        classifier.fit(X, y)                
        accuracy = cross_val_score(classifier, X, y, cv=5)              
        acc_mean = accuracy.mean()              
        print(f"分类器 {classifier_name} 准确率 {accuracy} 平均准确率 {acc_mean * 100:.2f}%")               
        results[classifier_name] = acc_mean             
    return results