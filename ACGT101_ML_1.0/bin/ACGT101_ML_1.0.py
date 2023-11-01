#!/usr/bin/env python
# coding: utf-8

from utils import read_config_file, print_params, read_data, Train_test
from model import build_Moudle, evaluate_model
from pre import pre_progress, preprogress
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import AdaBoostClassifier as ADA
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGBC
from figure import plot_roc, plot_pr, plot_confusion_matrix, plot_learning_curve, plot_lasso_path
from select_features import selected_feature, select_lasso
import argparse
import warnings
warnings.filterwarnings("ignore", category=Warning)
import os

def make_file():
    output="Output"
    if not os.path.exists(output):
        os.makedirs(output)
    output_path=os.path.abspath(output)
    return output_path


def main(output_path):
    # 1. 解析命令行参数
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="描述：高通量测序中的机器学习应用")

    # 添加命令行参数选项
    parser.add_argument("-p", "--parameters",
                        required=True, help="配置文件路径，必需选项。")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取解析后的参数值
    parameters = args.parameters

    # 读取配置文件
    params = read_config_file(parameters)

    # 打印参数值
    print_params(params)

    # 2. 加载数据
    x, y = read_data(params["file_path"])

    # 3. 数据预处理（数据归一化）
    X = pre_progress(x)

    # 4. 预训练
    results = preprogress(X, y)

    # 5. 特征选择
    if params["select"] == "RE":
        X = selected_feature(X, y, output_path=output_path)
    elif params["select"] == "Lasso":
        if "lasso_cv" in params:
            lasso_cv = int(params["lasso_cv"])
        else:
            lasso_cv = 10
        best_alpha = plot_lasso_path(X, y, lasso_cv=lasso_cv, output_path=output_path)
        print("best_alpha:", best_alpha)
        X = select_lasso(X, y, alpha=best_alpha, output_path=output_path)
    elif params["select"] == "None":
        print("No feature selection!")

    # 6. 划分训练集和测试集
    if "test_size" in params:
        test_size = float(params["test_size"])
    else:
        test_size = 0.2
    if 0 < test_size < 1:
        Xtrain, Xtest, Ytrain, Ytest = Train_test(
            X, y, test_size=test_size, random_state=int(params["random_state"]))
    else:
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    # 7. 构建模型
    if "cpu" in params:
        cpu = int(params["cpu"])
    else:
        cpu = 8
    if "min_child_weight" in params:
        min_child_weight = float(params["min_child_weight"])
    else:
        min_child_weight = 0.1
    if "max_depth_xgb" in params:
        max_depth_xgb = float(params["max_depth_xgb"])
    else:
        max_depth_xgb = 6
    if "reg_alpha" in params:
        reg_alpha = float(params["reg_alpha"])
    else:
        reg_alpha = None
    if "reg_lambda" in params:
        reg_lambda = float(params["reg_lambda"])
    else:
        reg_lambda = None
    if "kernel" in params:
        kernel = params["kernel"]
    else:
        kernel = 'rbf'
    if "C" in params:
        C = float(params["C"])
    else:
        C=1
    if "gamma" in params:
        gamma = params["gamma"]
    else:
        gamma = "scale"
    if "degree" in params:
        degree = int(params["degree"])
    else:
        degree = 3
    if "n_estimators" in params:
        n_estimators=int(params["n_estimators"])
    else:
        n_estimators=100
    if "max_depth_rf" in params:
        max_depth_rf = int(params["max_depth_rf"])
    else:
        max_depth_rf = None
    if "min_samples_split" in params:
        min_samples_split = int(params["min_samples_split"])
    else:
        min_samples_split = 2
    if "min_samples_leaf" in params:
        min_samples_leaf = int(params["min_samples_leaf"])
    else:
        min_samples_leaf = 1
    if "max_features" in params:
        max_features = params["max_features"]
    else:
        max_features = 'auto'
    if "solver" in params:
        solver = params["solver"]
    else:
        solver = 'svd'
    if "shrinkage" in params:
        shrinkage = params["shrinkage"]
    else:
        shrinkage = None
    if "n_components" in params:
        n_components = params["n_components"]
    else:
        n_components = None
    if "tol" in params:
        tol = params["tol"]
    else:
        tol = 1e-4
    if "max_depth_gbdt" in params:
        max_depth_gbdt = int(params["max_depth_gbdt"])
    else:
        max_depth_gbdt = 6
    if "learning_rate_gbdt" in params:
        learning_rate_gbdt = float(params["learning_rate_gbdt"])
    else:
        learning_rate_gbdt = 0.01
    if "n_estimators_gbdt" in params:
        n_estimators_gbdt = params["n_estimators_gbdt"]
    else:
        n_estimators_gbdt = 100
    if "subsample" in params:
        subsample = float(params["subsample"])
    else:
        subsample = 0.5
    if "max_depth_dt" in params:
        max_depth_dt = params["max_depth_dt"]
    else:
        max_depth_dt = 6
    if "min_samples_split_dt" in params:
        min_samples_split_dt = int(params["min_samples_split_dt"])
    else:
        min_samples_split_dt = 1
    if "min_samples_leaf_dt" in params:
        min_samples_leaf_dt = int(params["min_samples_leaf_dt"])
    else:
        min_samples_leaf_dt = 1
    if "weights" in params:
        weights = params["weights"]
    else:
        weights = 'distance'

    Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier = build_Moudle(Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, results=results,
                                                                                           max_depth_xgb=max_depth_xgb,
                                                                                           reg_alpha=reg_alpha,
                                                                                           reg_lambda=reg_lambda,
                                                                                           min_child_weight=min_child_weight,
                                                                                           kernel=kernel,
                                                                                           C=C,
                                                                                           n_estimators=n_estimators, 
                                                                                           max_depth_rf=max_depth_rf, 
                                                                                           min_samples_split=min_samples_split, 
                                                                                           min_samples_leaf=min_samples_leaf,
                                                                                           max_features=max_features,
                                                                                           gamma=gamma,
                                                                                           degree=degree,
                                                                                           cpu=cpu,
                                                                                           solver=solver, 
                                                                                           shrinkage=shrinkage, 
                                                                                           n_components=n_components, 
                                                                                           tol=tol,
                                                                                           max_depth_gbdt=max_depth_gbdt, 
                                                                                           learning_rate_gbdt=learning_rate_gbdt,
                                                                                           n_estimators_gbdt=n_estimators_gbdt,
                                                                                           subsample=subsample,
                                                                                           max_depth_dt=max_depth_dt, 
                                                                                           min_samples_split_dt=min_samples_split_dt, 
                                                                                           min_samples_leaf_dt=min_samples_leaf_dt,
                                                                                           weights=weights)

    # 8. 评估模型
    evaluate_model(best_classifier, Ytrain, Ytest, Ytrain_pred,
                   Ytest_pred, pred_score_more, pred_score_2, multi_class='ovo')

    # 9. 绘制ROC曲线
    plot_roc(Ytest, pred_score_2, pred_score_more, output_path=output_path)

    # 10. 绘制PR曲线
    if len(Ytest.unique()) < 3:
        plot_pr(Ytest, pred_score_2, pred_score_more, output_path=output_path)
    else:
        print("Multiclass classification detected. Skipping PR curve plot. ")

    # 11. 绘制混淆矩阵
    plot_confusion_matrix(Ytest, Ytest_pred, output_path=output_path)

    # 12. 绘制学习曲线
    if best_classifier == "逻辑回归":
        model = LR()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "决策树":
        model = DT()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "随机森林":
        model = RF()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "XGBoost":
        model = XGBC()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "支持向量机":
        model = SVC()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "GBDT":
        model = GBDT()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "ADAboost":
        model = ADA()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "朴素贝叶斯":
        model = GNB()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "线性判别分析":
        model = LDA()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)
    elif best_classifier == "k近邻":
        model = KNN()
        plot_learning_curve(model, Xtrain, Ytrain, Xtest, Ytest, output_path=output_path)


if __name__ == "__main__":
    output_path=make_file()
    main(output_path)
