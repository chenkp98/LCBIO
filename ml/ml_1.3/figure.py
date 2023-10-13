import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc
from pylab import mpl
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix as CM
import seaborn as sns

def plot_roc(Ytest,pred_score_2,pred_score_more,best_classifier):                  
    mpl.rcParams['font.sans-serif'] = ["SimSun"]                
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

    model_name=best_classifier
    plt.plot(FPR1, recall1, color='cyan',
            label='%s (area = %0.2f)'% (model_name,area1) ,linewidth=2.0)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--' ,linewidth=1.0)

    font={'family':'Times New Roman', 
      'weight':'normal',
      'size':20
     }
    font2={'family':'SimSun', 
        'weight':'normal',
        'size':15
        }
    plt.gca().set(xlim=(-0.05,1.05),ylim=(-0.05,1.05))
    plt.xticks(fontsize=12)                      
    plt.yticks(fontsize=12)
    plt.xlabel('False Positive Rate',fontdict=font)
    plt.ylabel('Recall',fontdict=font)
    plt.title('ROC curve',fontdict={'family':'Times New Roman',  'weight':'normal', 'size':25})
    plt.legend(loc="lower right",prop = font2,facecolor='w',edgecolor='w');
    plt.savefig("ROC_curve.jpg")




def plot_pr(Ytest,pred_score_2,pred_score_more,best_classifier):
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

    plt.plot(recall1,precision1,color="cyan",linewidth=2.0)

    font={'family':'Times New Roman', 
      'weight':'normal',
      'size':20
     }
    plt.xlabel("Recall",fontdict=font)
    plt.ylabel("Precision",fontdict=font)
    plt.xticks(fontsize=12)                      
    plt.yticks(fontsize=12)
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.title(' P-R curve' , fontdict={'family':'Times New Roman',  'weight':'normal', 'size':25})
    plt.legend(loc="lower right",prop = font,facecolor='w',edgecolor='w')
    plt.grid(False);
    plt.savefig("PR_curve.jpg")



def plot_confusion_matrix(Ytest, Ytest_pred):
    model = CM(Ytest, Ytest_pred)
    plt.figure(figsize=(8,8),dpi=300)
    sns.set(font="SimHei",font_scale=1.0)

    font={'family':'Times New Roman', 
    'weight':'normal',
      'size':25
    }

    ax=plt.gca()
    sns.heatmap(model, ax=ax, annot=True, cmap=plt.cm.GnBu, annot_kws={'size': 30}, cbar=False)
    ax.set_title("RF Confusion matrix",fontdict=font)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig("Confusion_Matrix")
