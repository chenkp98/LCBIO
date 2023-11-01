from mlmodel import ml_lr, ml_svm, ml_knn, ml_dt, ml_rf, ml_xgb, ml_lda, ml_gbdt, ml_ada, ml_gnb


def build_Moudle(Xtrain, Ytrain, Xtest, results, min_child_weight, max_depth_xgb, reg_alpha, reg_lambda, kernel, C, n_estimators, max_depth_rf, min_samples_split, min_samples_leaf, max_features,gamma, degree, cpu, solver, shrinkage, n_components, tol, max_depth_gbdt, learning_rate_gbdt, subsample, n_estimators_gbdt, min_samples_leaf_dt, max_depth_dt, min_samples_split_dt, weights):
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
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_lr(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "支持向量机":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_svm(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, kernel=kernel, C=C, gamma=gamma, degree=degree)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "k近邻":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_knn(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, weights=weights)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "决策树":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_dt(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, max_depth_dt=max_depth_dt, min_samples_split_dt=min_samples_split_dt, min_samples_leaf_dt=min_samples_leaf_dt)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "随机森林":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_rf(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, n_estimators=n_estimators, max_depth_rf=max_depth_rf, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "XGBoost":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_xgb(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, min_child_weight=min_child_weight, max_depth_xgb=max_depth_xgb, reg_alpha=reg_alpha, reg_lambda=reg_lambda, cpu=cpu)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "线性判别分析":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_lda(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, solver=solver, shrinkage=shrinkage, n_components=n_components, tol=tol, cpu=cpu)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "GBDT":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_gbdt(
            Xtrain=Xtrain, Ytrain=Ytrain, Xtest=Xtest, max_depth=max_depth_gbdt, learning_rate_gbdt=learning_rate_gbdt, n_estimators=n_estimators_gbdt, subsample=subsample, cpu=cpu)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "ADAboost":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_ada(
            Xtrain, Ytrain, Xtest, cpu=cpu)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier
    elif best_classifier == "朴素贝叶斯":
        Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2 = ml_gnb(
            Xtrain, Ytrain, Xtest)
        return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, best_classifier


def evaluate_model(best_classifier, Ytrain, Ytest, Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2, multi_class='ovo'):
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
        Cross_entropy_loss = log_loss(Ytest, Ytest_pred)
        auc = roc_auc_score(Ytest, pred_score_2)

    print(f"{best_classifier}模型结果\n训练集准确率: {train_accuracy}\n测试集准确率: {test_accuracy}\nF1分数: {F1_score}\n召回率: {Recall}\n精确率: {Precision}\n交叉熵损失: {Cross_entropy_loss}\nAUC值: {auc}")
