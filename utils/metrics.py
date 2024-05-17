import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def ACC(pred, true):
    return accuracy_score(true, pred)

def CONFUSION_MATRIX(pred, true):
    return confusion_matrix(true, pred)

def F1_SCORE(pred, true):
    return f1_score(true, pred, average='macro')


def ROC_AUC(true, pred_logits, folder_path, i=None):
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(true, pred_logits)
    # 计算AUC值
    roc_auc = roc_auc_score(true, pred_logits)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # 保存图片
    if i is not None:
        plt.savefig(folder_path + f"{i}.jpg")  # 保存图片到文件
    else:
        plt.savefig(folder_path + f"overall.jpg")

    plt.close()

    # 返回AUC值
    return roc_auc


def metric(pred, true, pred_logits, folder_path, i=None):
    acc = ACC(pred, true)
    conf_matrix = CONFUSION_MATRIX(pred, true)
    f1score = F1_SCORE(pred, true)
    auc = ROC_AUC(true, pred_logits, folder_path, i)

    return acc, conf_matrix, f1score, auc
