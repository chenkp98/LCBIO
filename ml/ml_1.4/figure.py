import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc
from pylab import mpl
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix as CM
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss


def plot_roc(Ytest,pred_score_2,pred_score_more):                  
    mpl.rcParams['font.sans-serif'] = ["Arial"]                
    mpl.rcParams["axes.unicode_minus"] = False 
    if len(Ytest.unique()) >2:
        FPR1,recall1,thresholds1=roc_curve(Ytest,pred_score_2,pos_label=1)
        area1=auc(Ytest,pred_score_more,multi_class='ovo')
    else:
        FPR1,recall1,thresholds1=roc_curve(Ytest,pred_score_2,pos_label=1)
        area1=auc(Ytest,pred_score_2)       

    plt.figure(figsize=(7,7),dpi=300,facecolor='white')
    ax=plt.gca()
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
            label='AUC = %0.2f' % area1 ,linewidth=2.0)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--' ,linewidth=1.0)

    font={'family':'Arial', 
      'weight':'normal',
      'size':20
     }
    font2={'family':'Arial', 
        'weight':'normal',
        'size':15
        }
    plt.gca().set(xlim=(-0.05,1.05),ylim=(-0.05,1.05))
    plt.xticks(fontsize=12)                      
    plt.yticks(fontsize=12)
    plt.xlabel('False Positive Rate',fontdict=font)
    plt.ylabel('Recall',fontdict=font)
    plt.title('ROC curve',fontdict={'family':'Arial',  'weight':'normal', 'size':25})
    plt.legend(loc="lower right",prop = font2,facecolor='w',edgecolor='w');
    plt.savefig("ROC_curve.jpg")




def plot_pr(Ytest,pred_score_2,pred_score_more):
    mpl.rcParams['font.sans-serif'] = ["SimSun"]                
    mpl.rcParams["axes.unicode_minus"] = False
    if len(Ytest.unique()) >2:
        precision1,recall1,_ = precision_recall_curve(Ytest,pred_score_more)
    else:
        precision1,recall1,_ = precision_recall_curve(Ytest,pred_score_2)


    plt.figure(figsize=(7,7),dpi=300,facecolor='w')
    ax=plt.gca()
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)

    plt.plot(recall1,precision1,color='blue',linewidth=2.0)

    font={'family':'Arial', 
      'weight':'normal',
      'size':20
     }
    plt.xlabel("Recall",fontdict=font)
    plt.ylabel("Precision",fontdict=font)
    plt.xticks(fontname='Arial',fontsize=12)                      
    plt.yticks(fontname='Arial',fontsize=12)
    plt.title(' P-R curve' , fontdict={'family':'Arial',  'weight':'normal', 'size':25})
    plt.legend(loc="lower right",prop = font,facecolor='w',edgecolor='w')
    plt.grid(False);
    plt.savefig("PR_curve.jpg")



def plot_confusion_matrix(Ytest, Ytest_pred):
    model = CM(Ytest, Ytest_pred)
    plt.figure(figsize=(6,6),dpi=300)
    sns.set(font="SimHei",font_scale=1.0)

    font={'family':'Arial', 
    'weight':'normal',
      'size':25
    }

    ax=plt.gca()
    sns.heatmap(model, ax=ax, annot=True, cmap=plt.cm.GnBu, annot_kws={'size': 50}, cbar=False)
    ax.set_title("RF Confusion matrix",fontdict=font)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlabel("Prediction label", fontdict=font)
    plt.ylabel("True label", fontdict=font)

    plt.savefig("Confusion_Matrix.jpg")



def plot_learning_curve(model, Xtrain, Ytrain, Xval, Yval):
    train_accs=[]
    val_accs=[]
    train_losses=[]
    val_losses=[]

    for i in range(10,len(Xtrain)+1,5):
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

    plt.figure(figsize=(8,5), dpi=300,facecolor='w')
    ax=plt.gca()
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_color('black')
    ax.spines['top'].set_linewidth(1.0)
    ax.spines['right'].set_color('black')
    ax.spines['right'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    plt.plot(range(10, len(Xtrain)+1, 5), train_accs, label='Train Accuracy', linewidth=2.0, linestyle='--', color='blue')
    plt.plot(range(10, len(Xtrain)+1, 5), val_accs, label='Val Accuracy',linewidth=2.0, linestyle='--', color='black')
    plt.plot(range(10, len(Xtrain)+1,5), train_losses, label='Train Loss',linewidth=2.0,linestyle='-', color='blue')
    plt.plot(range(10,len(Xtrain)+1, 5), val_losses, label='Val Loss', linewidth=2.0,linestyle='-', color='black')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Accuracy', fontdict=font)
    plt.xlabel('Training Examples', fontdict=font)
    plt.title("Training and Validation Loss & Accuracy", fontdict=font)
    plt.legend(prop={'size':10, 'family':'Arial','weight':'normal'}, facecolor='white');
    plt.savefig("learning_curve.jpg")


