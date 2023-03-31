import math
import numpy as np
from collections import Counter

class Taxonomy(object):
    """
    Taxonomy for Minority Class Examples
    - S is the number of Safe examples.
    - B is the number of Borderline examples.
    - R is the number of Rare examples.
    - O is the number of Outlier exaples.
    
    ------------
    Parameters
    - K: number of nearest neighbors

    ------------
    References 
    - [Napierala, K. and Stefanowski, J., 2016. Types of minority class examples and their influence on learning classifiers from imbalanced data. *Journal of Intelligent Information Systems*, 46, pp.563-597.](https://link.springer.com/article/10.1007/s10844-015-0368-1)
    """
    def __init__(self, K=5):
        self.K = K
        self.tax_nums = np.array([0, 0, 0, 0])
        self.tax_percentage = np.array([0, 0, 0, 0])
        self.labelled = None
        
    def fit(self, D, y):
        """
        Taxonomy for the Minority Class Examples for the distance matrix D with labels y.
        - S is the number of Safe examples.
        - B is the number of Borderline examples.
        - R is the number of Rare examples.
        - O is the number of Outlier examples.

        ------------
        Parameters
            D (np.array): data matrix of shape (N, M)
            y (np.array): labels of shape (N,)
            K (int): number of nearest neighbors to consider (default K=5)
        ------------
        Returns
            Dictionary with:
            - the counts of each type (key: "count")
            - the percentage of each type (key: "percentage")
            - the array with the leabels of each type (key: "target")
        """
        # initialize counters
        S = B = R = O = 0
        # labelled target array
        self.labelled = y.copy()

        # get the minority class
        minority_class = Counter(y).most_common()[-1][0]
        count_minority_class = Counter(y).most_common()[-1][1]
        # inifinite diagonal
        np.fill_diagonal(D, np.inf)
        # index where target = minprity_class
        idx_minority = np.where(y == minority_class)[0]

        distance = D[:,idx_minority].copy()
        idx_sorted =  np.argsort(distance, axis=0, kind='stable')

        # indices of K Nearest Neighbors
        idx_KNN = idx_sorted[:self.K]

        # target of K Nearest Neighbors
        target_KNN = y[idx_KNN]

        is_KNN_minority = target_KNN == minority_class

        # count minority class in KNN
        count_minority_KNN = np.sum(is_KNN_minority, axis=0)

        for i, sum_neighbors in enumerate(count_minority_KNN):
            if sum_neighbors >= math.floor(0.8*self.K): 
                S+=1
                self.labelled[idx_minority[i]] = 1
            elif sum_neighbors >= math.floor(0.5*self.K):
                B+=1
                self.labelled[idx_minority[i]] = 2
            elif sum_neighbors >= math.floor(0.2*self.K):
                # index of nearest neighbor of i
                idx_NN = idx_KNN[is_KNN_minority[:,i],i]
                i_NN = np.where(idx_minority == idx_NN)[0]
                # True Rare Examples
                if count_minority_KNN[i_NN] == 0 or (count_minority_KNN[i_NN] == 1 and idx_minority[i] in idx_KNN[:,i_NN]):
                    R+=1
                    self.labelled[idx_minority[i]] = 3
                # False Rare Examples => Borderline
                else:
                    B+=1
                    self.labelled[idx_minority[i]] = 2
            else:                               
                O+=1
                self.labelled[idx_minority[i]] = 4
            self.tax_nums = np.array([S, B, R, O])
            self.tax_percentage = self.tax_nums/count_minority_class*100
        return {'count' : self.tax_nums , 'percentage' : self.tax_percentage, 'target' : self.labelled}