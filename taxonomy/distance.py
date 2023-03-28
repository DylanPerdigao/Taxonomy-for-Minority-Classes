import numpy as np
import pandas as pd

class HVDM(object):
    """
    Heterogeneous Value Difference Metric

    ------------
    References 
    - [Wilson, D.R. and Martinez, T.R., 1997. Improved heterogeneous distance functions. *Journal of artificial intelligence research*, 6, pp.1-34.](https://www.jair.org/index.php/jair/article/view/10182)
    """
    def __init__(self, ddof=1, vdm_q=2):
        self.ddof = ddof # sample -> ddof=1
        self.vdm_q = vdm_q

    def normalized_vdm(self, data, target, valx, valy, a):
        """
        Compute the normalized value difference metric between x and y for a nominal attribute
        
        Parameters:
        x, y: values for the variable a
        sigma: standard deviation of the variable a
        a: boolean saying if it is nominal or not
        """
        C = np.unique(target)

        mask_valx = data[:,a] == valx
        mask_valy = data[:,a] == valy
        nax = mask_valx.sum()
        nay = mask_valy.sum()
        total = 0
        
        for c in C:
            mask_valx_c = mask_valx & (target == c)
            mask_valy_c = mask_valy & (target == c)
            naxc = mask_valx_c.sum()
            nayc = mask_valy_c.sum()
            #if nax != 0 and nay != 0:
                #total += np.abs((naxc/nax) - (nayc/nay)) ** 2
            total += abs((naxc/nax) - (nayc/nay))**self.vdm_q
        return total#**0.5
    

    def fit(self, data, target, nominal_attributes):
        n = data.shape[0]
        sigma = np.nanstd(data,axis=0, ddof=self.ddof)
        dist_matrix = np.zeros((n,n))
        for x in range(n-1):
            for y in range(x+1,n):
                    total = 0
                    for a in range(data.shape[1]):
                        x_a = data[x,a]
                        y_a = data[y,a]
                        if pd.isna(x_a) or pd.isna(y_a): # pd.isna supports category type
                            d = 1
                        elif x_a == y_a:
                            d = 0
                        elif nominal_attributes[a]:
                            d = self.normalized_vdm(data,target,x_a,y_a,a)#**2
                        elif 4*sigma[a] == 0:
                            d = 1
                        else:
                            d = (np.abs(float(x_a)-float(y_a))/(4*sigma[a]))**2
                        total += d
                    dist_matrix[x,y] = dist_matrix[y,x] = np.sqrt(total)
        return dist_matrix
