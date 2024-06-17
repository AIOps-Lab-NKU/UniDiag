import math
import numpy as np
from sklearn import metrics
import pandas as pd
from scipy.optimize import linear_sum_assignment as linear_assignment
#import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

def NMI(A,B):
    # samples numbers:
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    # calculation mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)    # output the indices of elements that meet certain conditions.
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)   # Find the intersection of two arrays.
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual Information.
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def ACC(Y, Y_pred):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    total = 0
    for i in range(len(ind[0])):
        total += w[ind[0][i], ind[1][i]]
    return total * 1.0 / Y_pred.size


# Euclidean,Manhatton,Chebyshev,Cos,Square,Wasserstein,SBD,Pearsonr
# AgglomerativeClustering,DBSCAN,KMeans
def score_report(stat_path,report_path):
    result = pd.read_csv(stat_path)
    y_true = result['label']
    y_pred = result['result']

    print('NMI', NMI(y_true, y_pred))
    print('ACC', ACC(y_true, y_pred))
    print('Weighted precision', precision_score(y_true, y_pred, average='weighted'))
    print('Weighted recall', recall_score(y_true, y_pred, average='weighted'))
    print('Weighted f1-score', f1_score(y_true, y_pred, average='weighted'))
    
    f = open(report_path,'w')
    f.write('NMI {}\n'.format(NMI(y_true, y_pred)))
    f.write('ACC {}\n'.format(ACC(y_true, y_pred)))
    f.write('Weighted precision {}\n'.format(precision_score(y_true, y_pred, average='weighted')))
    f.write('Weighted recall {}\n'.format(recall_score(y_true, y_pred, average='weighted')))
    f.write('Weighted f1-score {}\n'.format(f1_score(y_true, y_pred, average='weighted')))
    f.close()
