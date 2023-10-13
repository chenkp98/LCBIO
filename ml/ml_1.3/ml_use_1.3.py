#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", category=Warning)
import argparse
from utils import read_config_file,print_params,ReadD,pre_progress,PreProgress,selected_feature,Train_test,build_Moudle,evaluate_model
from figure import plot_roc, plot_pr, plot_confusion_matrix

def main():
    # 1. 解析命令行参数
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="描述：高通量测序中的机器学习应用")

    # 添加命令行参数选项
    parser.add_argument("-p", "--parameters", required=True, help="配置文件路径，必需选项。")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取解析后的参数值
    parameters = args.parameters

    # 读取配置文件
    params = read_config_file(parameters)

    # 打印参数值
    print_params(params)

    # 2. 加载数据
    x, y = ReadD(params["file_path"])

    # 3. 数据预处理（数据归一化）
    X = pre_progress(x)

    # 4. 预训练
    results = PreProgress(X, y)

    # 5. 特征选择
    if params["select"] == "True":
        X = selected_feature(X, y)

    # 6. 划分训练集和测试集
    test_size = float(params["test_size"])
    if 0 < test_size < 1:
        Xtrain, Xtest, Ytrain, Ytest = Train_test(X, y, test_size=test_size, random_state=int(params["random_state"]))
    else:
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    # 7. 构建模型
    if "min_child_weight" in params:
        min_child_weight=float(params["min_child_weight"])
    else:
        min_child_weight = 0.1
    if "max_depth" in params:
        max_depth=float(params["max_depth"])
    else:
        max_depth = 6
    if "reg_alpha" in params:
        reg_alpha=float(params["reg_alpha"])
    else:
        reg_alpha = None
    if "reg_lambda" in params:
        reg_lambda=float(params["reg_lambda"])
    else:
        reg_lambda = None
    if "kernel" in params:
        kernel=params["kernel"]
    else:
        kernel = 'rbf'
    Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier = build_Moudle(X, y, Xtrain, Ytrain, Xtest, Ytest,results,
                                                                           min_child_weight=min_child_weight,
                                                                           max_depth=max_depth,
                                                                           reg_alpha=reg_alpha,
                                                                           reg_lambda=reg_lambda,
                                                                            kernel=kernel)

    # 8. 评估模型
    evaluate_model(best_classifier,Ytrain,Ytest,Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,multi_class='ovo')

    # 9. 绘制ROC曲线
    plot_roc(Ytest,pred_score_2,pred_score_more,best_classifier)

    # 10. 绘制PR曲线
    if len(Ytest.unique()) < 3:
        plot_pr(Ytest, pred_score_2,pred_score_more,best_classifier)
    else:
        print("Multiclass classification detected. Skipping PR curve plot. ")

    
    # 11. 绘制混淆矩阵
    plot_confusion_matrix(Ytest, Ytest_pred)

if __name__ == "__main__":
    main()


"""
if __name__ == "__main__":
    main()
    x,y=ReadD("D:/chenh/李/data_li.xlsx")
    X = pre_progress(x)
    results = PreProgress(X,y)
    X = selected_feature(X,y)
    Xtrain, Xtest, Ytrain, Ytest = Train_test(X,y,0.2,420)
    Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier = build_Moudle(X, y, Xtrain,Ytrain, Xtest, Ytest,0.2,0.3,10,1,kernel='rbf')
    evaluate_model(best_classifier,Ytrain,Ytest,Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2)
"""





