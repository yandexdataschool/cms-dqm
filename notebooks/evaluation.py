from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

import numpy as np
import matplotlib.pyplot as plt


#fixed recall values
rec_values = [0.8, 0.9, 0.95, 0.99]


# precision at 10 or P@10 measures classification performance, 
# being the fraction of the top 10 scored instances which are actually anomalous.
def Pat10(y_test, y_pred, n=10):
    ind = np.argpartition(y_pred, -n)[-n:]
    return np.mean(y_test[ind])

def perfomance(y_test, y_pred, sample_weight=None, n=10):
    print ("P@"+str(n), Pat10(y_test, y_pred, n))

    fig, axes = plt.subplots(nrows=3, figsize=(6, 15))
    
    ax = axes[0]
    ax.grid(True)
    precision, recall, _ = precision_recall_curve(y_test,y_pred)
    
    print ("recalls_values",rec_values)
    prec_values = []
    for v in rec_values:
        prec_values.append(max(precision[recall > v]))
    print ("precision_values", prec_values)
    
    print ("average_precision_score", average_precision_score(y_test, y_pred, sample_weight=sample_weight))
    print ("roc_auc_score", roc_auc_score(y_test, y_pred))
        

    ax.step(recall, precision, color='b', alpha=0.2,
             where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title('2-class Precision-Recall curve: AP={0:0.3f}'.format(
              average_precision_score(y_test, y_pred, sample_weight=sample_weight)), fontsize=25)
    
    ax = axes[1]
    ax.grid(True)
    fpr, tpr, _ = roc_curve(y_test, y_pred, sample_weight=sample_weight)
    ax.plot(fpr, tpr)
    ax.set_title('ROC curve: roc_auc ={0:0.3f}'.format(
                 roc_auc_score(y_test, y_pred)), fontsize=25)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('FPR', fontsize=10)
    ax.set_ylabel('TPR', fontsize=10)
    
    ax = axes[2]
    bad_test = np.sum(y_test)
    good_test = len(y_test)-np.sum(y_test)
    ax.plot(sorted(y_pred[np.where( y_test == 0.)[0]], reverse=True), np.arange(good_test)/good_test*100, label = "good")
    ax.plot(sorted(y_pred[np.where( y_test == 1.)[0]]), np.arange(bad_test)/bad_test*100, label = "bad")
    ax.set_title('Predicted proba', fontsize=25)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    fig.subplots_adjust(hspace=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return precision, recall