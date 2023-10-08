#!/usr/bin/env python
# coding: utf-8


import argparse
from pre import read_config_file,print_params,ReadD,pre_progress,PreProgress,selected_feature,Train_test,build_Moudle,evaluate_model


def main():
    # 1. 解析命令行参数
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="描述：高通量测序中的机器学习应用")

    # 添加命令行参数选项
    parser.add_argument("-c", "--config_file", required=True, help="配置文件路径，必需选项。")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取解析后的参数值
    config_file = args.config_file

    # 读取配置文件
    params = read_config_file(config_file)

    # 打印参数值
    print_params(params)

    # 2. 加载数据
    x, y = ReadD(params["file_path"])

    # 3. 数据预处理（数据归一化）
    X = pre_progress(x)

    # 4. 预训练
    results = PreProgress(X, y)

    # 5. 特征选择
    if params["select"]:
        X = selected_feature(X, y)

    # 6. 划分训练集和测试集
    test_size = float(params["test_size"])
    if 0 < test_size < 1:
        Xtrain, Xtest, Ytrain, Ytest = Train_test(X, y, test_size=test_size, random_state=int(params["random_state"]))
    else:
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    # 7. 构建模型
    Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = build_Moudle(X, y, Xtrain, Ytrain, Xtest, Ytest)

    # 8. 评估模型
    evaluate_model(Ytrain,Ytest,Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2,multi_class='ovo')





if __name__ == "__main__":
    main()
    x,y=ReadD("D:/data/11.xlsx")
    X = pre_progress(x)
    results = PreProgress(X,y)
    X = selected_feature(X,y)
    Xtrain, Xtest, Ytrain, Ytest = Train_test(X,y,0.2,420)
    Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2 = build_Moudle(X, y, Xtrain,Ytrain, Xtest, Ytest)
    evaluate_model(Ytrain,Ytest,Ytrain_pred,Ytest_pred,pred_score_more,pred_score_2)




