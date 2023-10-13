import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression as LR, Lasso as Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DT
from xgboost import XGBClassifier as XGB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF, GradientBoostingClassifier as GBDT, AdaBoostClassifier as ADA
from sklearn.naive_bayes import GaussianNB as GNB
from mlmodel import ml_lr, ml_la, ml_svm, ml_knn, ml_dt, ml_rf, ml_xgb, ml_lda, ml_gbdt, ml_ada, ml_gnb



def read_config_file(config_file):
    params = {}
    with open(config_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=")
                params[key.strip()] = value.strip()
    return params

def print_params(params):
    print("参数：")
    for key, value in params.items():
        print(f"{key}: {value}")

def ReadD(file_path):
    file_extension = file_path.split(".")[-1].lower()
    if file_extension == "csv":
        data = pd.read_csv(file_path)
    elif file_extension == "xlsx":
        data = pd.read_excel(file_path)
    elif file_extension == "pickle":
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
        # 如果pickle文件中保存的是DataFrame对象，则直接返回
            return data.iloc[:, 1:-1], data.iloc[:, -1]
        else:
        # 如果pickle文件中保存的是其他对象，则根据具体情况进行处理
        # 这里只是一个示例，你可以根据你的数据类型进行适当的处理
            return pd.DataFrame(data), None
    else:
        raise ValueError("不支持的文件类型：{}".format(file_extension))

    x = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    return x, y

def pre_progress(x):                
    from sklearn.preprocessing import MinMaxScaler              
    scaler = MinMaxScaler().fit_transform(x)  
    X = pd.DataFrame(scaler,columns=x.columns)           
    return X                


    

def PreProgress(X, y):
    classifiers = {             
            "逻辑回归": LR(),               
            "Lasso回归": Lasso(),               
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


def selected_feature(X, y):             
    import matplotlib.pyplot as plt             
    from pylab import mpl               
    mpl.rcParams['font.sans-serif'] = ["SimSun"]                
    mpl.rcParams["axes.unicode_minus"] = False              
    from sklearn.feature_selection import RFE               
    np.random.seed(420)             

    rf=RF().fit(X, y)               
    svm = SVC(kernel='linear').fit(X,y)             

    fontdict_title={'fontsize':20,              
            'family': 'SimSun',              
            'weight': 'normal'               
    }               

    score_rf=[]             
    for i in range(1,X.shape[1]+1,1):               
        X_wrapper_rf = RFE(rf,n_features_to_select=i).fit_transform(X, y)               
        once_rf = cross_val_score(rf, X_wrapper_rf, y, cv=5).mean()             
        score_rf.append(once_rf)                
    plt.figure(figsize=[20,5], dpi=200)             
    plt.plot(range(1,X.shape[1]+1,1),score_rf)              
    plt.xticks(range(1,X.shape[1]+1,1),fontsize=10)             
    plt.yticks(fontsize=15)             
    plt.title("随机森林递归法寻找最优特征个数",fontdict=fontdict_title);                
    plt.savefig("随机森林递归法学习曲线.jpg")               

    score_svm = []              
    for i in range(1,X.shape[1]+1,1):               
        X_wrapper_svm = RFE(svm,n_features_to_select=i).fit_transform(X, y)             
        once_svm = cross_val_score(rf, X_wrapper_svm, y, cv=5).mean()               
        score_svm.append(once_svm)              
    plt.figure(figsize=[20,5],dpi=200)              
    plt.plot(range(1,X.shape[1]+1,1),score_svm)             
    plt.xticks(range(1,X.shape[1]+1,1),fontsize=10)             
    plt.yticks(fontsize=15)             
    plt.title("支持向量机递归法寻找最优特征个数", fontdict=fontdict_title);             
    plt.savefig("支持向量机递归法学习曲线.jpg")                

    num_feature_rf = score_rf.index(max(score_rf))+1                
    num_feature_svm = score_svm.index(max(score_svm))+1             
    selectors = [               
    (RFE(rf, n_features_to_select=num_feature_rf), "随机森林递归特征法"),               
    (RFE(svm, n_features_to_select=num_feature_svm),"支持向量机递归特征法")             
    ]               

    selected_features_all = set()               
    for selector, selector_name in selectors:               
        select_model = selector.fit(X, y)               
        selected_features = X.columns[select_model.get_support()]               
        selected_features_str = ','.join(selected_features)             
        selected_features_all.update(selected_features)             
        print(f"{selector_name}筛选出的特征：{selected_features_str}")              
    print(f"被选出的所有特征 {selected_features_all}")              
    X = X[list(selected_features_all)]                
    return X                


from sklearn.model_selection import train_test_split
def Train_test(X, y, test_size, random_state):                       
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)               
    return Xtrain, Xtest, Ytrain, Ytest             


def build_Moudle(X, y, Xtrain,Ytrain,Xtest,Ytest, results, min_child_weight,max_depth,reg_alpha,reg_lambda, kernel):                
    best_classifier = None              
    max_accuracy = 0      
    for classifier_name, accuracy in results.items():               
        if accuracy > max_accuracy:             
            best_classifier = classifier_name               
            max_accuracy = accuracy             
        elif accuracy == max_accuracy:              
        # 如果有多个分类器具有相同的最高准确率，可以在此处进行适当的处理                
            pass                
    print("准确率最高的分类器：", best_classifier)              
    # 使用最高准确率的分类器进行建模                
    if best_classifier == "逻辑回归":               
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_lr(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier              
    elif best_classifier =="Lasso回归":             
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_la(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier              
    elif best_classifier == "支持向量机":               
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_svm(Xtrain,Ytrain,Xtest,Ytest,kernel)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier             
    elif best_classifier == "k近邻":                
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_knn(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier             
    elif best_classifier == "决策树":               
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_dt(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier              
    elif best_classifier == "随机森林":             
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_rf(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier              
    elif best_classifier == "XGBoost":              
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_xgb(Xtrain,Ytrain,Xtest, min_child_weight, max_depth, reg_alpha,reg_lambda=reg_lambda)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier             
    elif best_classifier == "线性判别分析":             
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_lda(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier             
    elif best_classifier == "GBDT":             
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_gbdt(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier                
    elif best_classifier == "ADAboost":             
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_ada(Xtrain,Ytrain,Xtest,Ytest)
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier             
    elif best_classifier == "朴素贝叶斯":               
        Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = ml_gnb(Xtrain,Ytrain,Xtest,Ytest)             
        return Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,best_classifier              

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier                


def evaluate_model(best_classifier,Ytrain,Ytest,Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,multi_class='ovo'):             
    from sklearn.metrics import precision_score as precision, recall_score as recall, f1_score as F1, accuracy_score as accuracy, roc_auc_score, log_loss               
    train_accuracy = accuracy(Ytrain, Ytrain_pred)              
    test_accuracy = accuracy(Ytest, Ytest_pred)             

    if len(Ytest.unique()) > 2:             
        F1_score = F1(Ytest, Ytest_pred, average='micro')               
        Recall = recall(Ytest, Ytest_pred, average='micro')             
        Precision = precision(Ytest, Ytest_pred, average='micro')               
        Cross_entropy_loss = log_loss(Ytest, pred_score_more)               
        auc = roc_auc_score(Ytest, pred_score_more, multi_class=multi_class)                
    else:               
        F1_score = F1(Ytest, Ytest_pred)                
        Recall = recall(Ytest, Ytest_pred)              
        Precision = precision(Ytest, Ytest_pred)                
        Cross_entropy_loss = log_loss(Ytest, Ytest_pred )               
        auc = roc_auc_score(Ytest,pred_score_2)             

    print(f"{best_classifier}模型结果\n训练集准确率: {train_accuracy}\n测试集准确率: {test_accuracy}\nF1分数: {F1_score}\n召回率: {Recall}\n精确率: {Precision}\n交叉熵损失: {Cross_entropy_loss}\nAUC值: {auc}")               