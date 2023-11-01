import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc
from pylab import mpl
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix as CM
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV


def plot_roc(Ytest, pred_score_2, pred_score_more, output_path):
    mpl.rcParams['font.sans-serif'] = ["Arial"]
    mpl.rcParams["axes.unicode_minus"] = False
    if len(Ytest.unique()) > 2:
        FPR1, recall1, thresholds1 = roc_curve(
            Ytest, pred_score_2, pos_label=1)
        area1 = auc(Ytest, pred_score_more, multi_class='ovo')
    else:
        FPR1, recall1, thresholds1 = roc_curve(
            Ytest, pred_score_2, pos_label=1)
        area1 = auc(Ytest, pred_score_2)

    plt.figure(figsize=(7, 7), dpi=300, facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)

    plt.plot(FPR1, recall1, color='blue',
             label='AUC = %0.2f' % area1, linewidth=2.0)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.0)

    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 20
            }
    font2 = {'family': 'Arial',
             'weight': 'normal',
             'size': 15
             }
    plt.gca().set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('False Positive Rate', fontdict=font)
    plt.ylabel('Recall', fontdict=font)
    plt.title('ROC curve', fontdict={
              'family': 'Arial',  'weight': 'normal', 'size': 20})
    plt.legend(loc="lower right", prop=font2, facecolor='w', edgecolor='w')
    plt.savefig(f"{output_path}/ROC_curve.jpg")


def plot_pr(Ytest, pred_score_2, pred_score_more, output_path):
    mpl.rcParams['font.sans-serif'] = ["SimSun"]
    mpl.rcParams["axes.unicode_minus"] = False
    if len(Ytest.unique()) > 2:
        precision1, recall1, _ = precision_recall_curve(Ytest, pred_score_more)
    else:
        precision1, recall1, _ = precision_recall_curve(Ytest, pred_score_2)

    plt.figure(figsize=(7, 7), dpi=300, facecolor='w')
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)

    plt.plot(recall1, precision1, color='blue', linewidth=2.0)

    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 20
            }
    plt.xlabel("Recall", fontdict=font)
    plt.ylabel("Precision", fontdict=font)
    plt.xticks(fontname='Arial', fontsize=12)
    plt.yticks(fontname='Arial', fontsize=12)
    plt.title(' P-R curve',
              fontdict={'family': 'Arial',  'weight': 'normal', 'size': 20})
    plt.legend(loc="lower right", prop=font, facecolor='w', edgecolor='w')
    plt.grid(False)
    plt.savefig(f"{output_path}/PR_curve.jpg")


def plot_confusion_matrix(Ytest, Ytest_pred, output_path):
    model = CM(Ytest, Ytest_pred)
    plt.figure(figsize=(5, 5), dpi=300)
    sns.set(font="SimHei", font_scale=1.0)

    font = {'family': 'Arial',
            'weight': 'normal',
            'size': 15
            }

    ax = plt.gca()
    sns.heatmap(model, ax=ax, annot=True, cmap=plt.cm.GnBu,
                annot_kws={'size': 50}, cbar=False)
    ax.set_title("RF Confusion matrix", fontdict=font)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlabel("Prediction label", fontdict=font)
    plt.ylabel("True label", fontdict=font)

    plt.savefig(f"{output_path}/Confusion_Matrix.jpg")


def plot_learning_curve(model, Xtrain, Ytrain, Xval, Yval, output_path):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    for i in range(10, len(Xtrain)+1, 5):
        model.fit(Xtrain[:i], Ytrain[:i])

        train_pred = model.predict(Xtrain[:i])
        val_pred = model.predict(Xval)

        train_acc = accuracy_score(Ytrain[:i], train_pred)
        val_acc = accuracy_score(Yval, val_pred)
        train_loss = log_loss(Ytrain[:i], train_pred)
        val_loss = log_loss(Yval, val_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    font = {
        'size': 15,
        'family': 'Arial',
        'weight': 'normal'
    }

    plt.figure(figsize=(8, 5), dpi=300, facecolor='w')
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    
    plt.plot(range(10, len(Xtrain)+1, 5), train_losses,
             label='Train Loss', linewidth=2.0, linestyle='-', color='blue')
    plt.plot(range(10, len(Xtrain)+1, 5), val_losses, label='Val Loss',
             linewidth=2.0, linestyle='-', color='black')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Loss', fontdict=font)
    plt.xlabel('Training Examples', fontdict=font)
    plt.title("Training and Validation Loss", fontdict=font)
    plt.legend(prop={'size': 10, 'family': 'Arial',
               'weight': 'normal'}, facecolor='white')
    plt.savefig(f"{output_path}/loss_learning_curve.jpg")
    
    plt.figure(figsize=(8, 5), dpi=300, facecolor='w')
    ax = plt.gca()
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    plt.plot(range(10, len(Xtrain)+1, 5), train_accs,
             label='Train Accuracy', linewidth=2.0, linestyle='--', color='blue')
    plt.plot(range(10, len(Xtrain)+1, 5), val_accs, label='Val Accuracy',
             linewidth=2.0, linestyle='--', color='black')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Accuracy', fontdict=font)
    plt.xlabel('Training Examples', fontdict=font)
    plt.title("Training and Validation Accuracy", fontdict=font)
    plt.legend(prop={'size': 10, 'family': 'Arial',
               'weight': 'normal'}, facecolor='white')
    plt.savefig(f"{output_path}/accuracy_learning_curve.jpg")




def plot_lasso_path(X, y, lasso_cv, output_path):
    lasso = Lasso(alpha=0)
    coefs = []
    alphas = np.logspace(-5, 1, 100)
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        lasso.set_params(alpha=alpha)
        lasso.fit(X, y)
        coefs.append(lasso.coef_)

        scores = cross_val_score(lasso, X, y, cv=lasso_cv)
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha

    font = {
        "family": "Arial",
        "size": 15,
        "weight": "normal"
    }
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(alphas, coefs)
    plt.xscale("log")
    plt.xlabel("Alpha", fontdict=font)
    plt.ylabel("Coefficients", fontdict=font)
    plt.title("Lasso Coefficients", fontdict=font)
    plt.grid(True)
    plt.savefig(f"{output_path}/lasso_path.jpg")

    lasso_cv = LassoCV(cv=10)
    lasso_cv.fit(X, y)

    alphas = lasso_cv.alphas_
    mse_mean = np.mean(lasso_cv.mse_path_, axis=1)
    mse_std = np.std(lasso_cv.mse_path_, axis=1)

    best_alpha = lasso_cv.alpha_
    se = mse_std[np.where(alphas == best_alpha)]
    alpha_1se = alphas[np.where(mse_mean <= mse_mean.min() + se)][0]
    alpha_2se = alphas[np.where(mse_mean <= mse_mean.min() + 2 * se)][0]

    font = {
        "size": 15,
        "family": "Arial",
        "weight": "normal"
    }
    plt.figure(figsize=(8, 5), dpi=300)
    plt.errorbar(alphas, mse_mean, yerr=mse_std, fmt='o', capsize=3,
                 label='Mean ± Std', markersize=4, color="red", linewidth=0.5, ecolor="gray")
    plt.axvline(x=alpha_1se, color='grey', linestyle='--', label='1 SE')
    plt.axvline(x=alpha_2se, color='grey', linestyle='--', label='2 SE')
    plt.xlabel('Alpha', fontdict=font)
    plt.ylabel('Mean Squared Error', fontdict=font)
    plt.title('Lasso Cross-Validation', fontdict=font)
    plt.xscale('log')
    plt.savefig(f"{output_path}/lasso_cv.jpg")

    return best_alpha
