def ml_xgb(Xtrain, Ytrain, Xtest, min_child_weight, max_depth_xgb, reg_alpha, reg_lambda, cpu):
    from xgboost import XGBClassifier as XGBC
    XGB = XGBC(min_child_weight=min_child_weight, max_depth=max_depth_xgb,
               reg_alpha=reg_alpha, reg_lambda=reg_lambda, n_jobs=cpu).fit(Xtrain, Ytrain)
    Ytrain_pred = XGB.predict(Xtrain)
    Ytest_pred = XGB.predict(Xtest)
    pred_score_2 = XGB.predict_proba(Xtest)[:, 1]
    pred_score_more = XGB.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_svm(Xtrain, Ytrain, Xtest, kernel, C, gamma, degree):
    from sklearn.svm import SVC
    SVM = SVC(probability=True, class_weight='balanced',
              kernel=kernel, C=C, gamma=gamma, degree=degree).fit(Xtrain, Ytrain)
    Ytrain_pred = SVM.predict(Xtrain)
    Ytest_pred = SVM.predict(Xtest)
    pred_score_2 = SVM.predict_proba(Xtest)[:, 1]
    pred_score_more = SVM.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_lr(Xtrain, Ytrain, Xtest):
    from sklearn.linear_model import LogisticRegression as LR
    lr = LR().fit(Xtrain, Ytrain)
    Ytrain_pred = lr.predict(Xtrain)
    Ytest_pred = lr.predict(Xtest)
    pred_score_2 = lr.predict_proba(Xtest)[:, 1]
    pred_score_more = lr.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_knn(Xtrain, Ytrain, Xtest, weights):
    from sklearn.neighbors import KNeighborsClassifier as KNN
    knn = KNN(weights=weights).fit(Xtrain, Ytrain)
    Ytrain_pred = knn.predict(Xtrain)
    Ytest_pred = knn.predict(Xtest)
    pred_score_2 = knn.predict_proba(Xtest)[:, 1]
    pred_score_more = knn.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_dt(Xtrain, Ytrain, Xtest, max_depth_dt, min_samples_split_dt, min_samples_leaf_dt):
    from sklearn.tree import DecisionTreeClassifier as DT
    dt = DT(max_depth=max_depth_dt, min_samples_split=min_samples_split_dt, min_samples_leaf=min_samples_leaf_dt, random_state=420).fit(Xtrain, Ytrain)
    Ytrain_pred = dt.predict(Xtrain)
    Ytest_pred = dt.predict(Xtest)
    pred_score_2 = dt.predict_proba(Xtest)[:, 1]
    pred_score_more = dt.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_rf(Xtrain, Ytrain, Xtest, n_estimators, max_depth_rf, min_samples_split, min_samples_leaf, max_features):
    from sklearn.ensemble import RandomForestClassifier as RF
    rf = RF(n_estimators=n_estimators, max_depth=max_depth_rf, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=420).fit(Xtrain, Ytrain)
    Ytrain_pred = rf.predict(Xtrain)
    Ytest_pred = rf.predict(Xtest)
    pred_score_2 = rf.predict_proba(Xtest)[:, 1]
    pred_score_more = rf.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_lda(Xtrain, Ytrain, Xtest, solver, shrinkage, n_components, tol, cpu):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(solver=solver, shrinkage=shrinkage,
              n_components=n_components, tol=tol, n_jobs=cpu).fit(Xtrain, Ytrain)
    Ytrain_pred = lda.predict(Xtrain)
    Ytest_pred = lda.predict(Xtest)
    pred_score_2 = lda.predict_proba(Xtest)[:, 1]
    pred_score_more = lda.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_gbdt(Xtrain, Ytrain, Xtest, max_depth_gbdt, learning_rate_gbdt, n_estimators_gbdt, subsample, cpu):
    from sklearn.ensemble import GradientBoostingClassifier as GBDT
    gbdt = GBDT(max_depth=max_depth_gbdt, learning_rate=learning_rate_gbdt,
                n_estimators=n_estimators_gbdt, subsample=subsample, n_jobs=cpu).fit(Xtrain, Ytrain)
    Ytrain_pred = gbdt.predict(Xtrain)
    Ytest_pred = gbdt.predict(Xtest)
    pred_score_2 = gbdt.predict_proba(Xtest)[:, 1]
    pred_score_more = gbdt.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_ada(Xtrain, Ytrain, Xtest, cpu):
    from sklearn.ensemble import AdaBoostClassifier as ADA
    ada = ADA(n_jobs=cpu).fit(Xtrain, Ytrain)
    Ytrain_pred = ada.predict(Xtrain)
    Ytest_pred = ada.predict(Xtest)
    pred_score_2 = ada.predict_proba(Xtest)[:, 1]
    pred_score_more = ada.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2


def ml_gnb(Xtrain, Ytrain, Xtest):
    from sklearn.naive_bayes import GaussianNB as GNB
    gnb = GNB().fit(Xtrain, Ytrain)
    Ytrain_pred = gnb.predict(Xtrain)
    Ytest_pred = gnb.predict(Xtest)
    pred_score_2 = gnb.predict_proba(Xtest)[:, 1]
    pred_score_more = gnb.predict_proba(Xtest)

    return Ytrain_pred, Ytest_pred, pred_score_more, pred_score_2
