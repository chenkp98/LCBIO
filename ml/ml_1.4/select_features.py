import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso


def selected_feature(X, y):
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ["SimSun"]
    mpl.rcParams["axes.unicode_minus"] = False
    from sklearn.feature_selection import RFE
    np.random.seed(420)

    rf = RF().fit(X, y)
    svm = SVC(kernel='linear').fit(X, y)

    fontdict_title = {'fontsize': 20,
                      'family': 'SimSun',
                      'weight': 'normal'
                      }

    score_rf = []
    for i in range(1, X.shape[1]+1, 1):
        X_wrapper_rf = RFE(rf, n_features_to_select=i).fit_transform(X, y)
        once_rf = cross_val_score(rf, X_wrapper_rf, y, cv=5).mean()
        score_rf.append(once_rf)
    plt.figure(figsize=[20, 5], dpi=300)
    plt.plot(range(1, X.shape[1]+1, 1), score_rf)
    plt.xticks(range(1, X.shape[1]+1, 1), fontsize=10)
    plt.yticks(fontsize=15)
    plt.title("随机森林递归法寻找最优特征个数", fontdict=fontdict_title)
    plt.savefig("randomforest_features_curve.jpg")

    score_svm = []
    for i in range(1, X.shape[1]+1, 1):
        X_wrapper_svm = RFE(svm, n_features_to_select=i).fit_transform(X, y)
        once_svm = cross_val_score(rf, X_wrapper_svm, y, cv=5).mean()
        score_svm.append(once_svm)
    plt.figure(figsize=[20, 5], dpi=300)
    plt.plot(range(1, X.shape[1]+1, 1), score_svm)
    plt.xticks(range(1, X.shape[1]+1, 1), fontsize=10)
    plt.yticks(fontsize=15)
    plt.title("支持向量机递归法寻找最优特征个数", fontdict=fontdict_title)
    plt.savefig("SVM_features_curve.jpg")

    num_feature_rf = score_rf.index(max(score_rf))+1
    num_feature_svm = score_svm.index(max(score_svm))+1
    selectors = [
        (RFE(rf, n_features_to_select=num_feature_rf), "随机森林递归特征法"),
        (RFE(svm, n_features_to_select=num_feature_svm), "支持向量机递归特征法")
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


def select_lasso(X, y, alpha=0.05):
    la = Lasso(alpha)
    la.fit(X, y)
    scores = sorted([*zip(X.columns, la.coef_.ravel())],
                    key=lambda s: abs(s[1]), reverse=True)

    keys = []
    values = []
    for score in scores:
        if score[1] != 0:
            key = score[0]
            value = score[1]
            keys.append(key)
            values.append(value)

    font = {
        'size': 20,
        'family': 'Arial',
        'weight': 'normal'
    }
    plt.figure(figsize=(len(keys)-1,7),dpi=300)
    plt.bar(x=keys, height=values, color='blue')
    plt.xlabel("Feature", fontdict=font)
    plt.ylabel("Value", fontdict=font)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Feature Importance", fontdict=font)
    plt.savefig("select_lasso.jpg")

    X = X[keys]
    return X
