import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

class NMTF():

  def __init__(self,
         R12=None,  R12_train=None,
         R23=None, R24=None, R34=None,
         L3 = None, L4 = None,
         init_method = None, init_params = None, 
         train_indices_R12=None, test_indices_R12=None):

    self.R12 = R12
    self.R12_train = R12_train
    self.R23 = R23
    self.R24 = R24
    self.R34 = R34
    self.L3 = L3
    self.L4 = L4
    self.init_method = init_method
    self.K = init_params
    self.train_indices_R12 = train_indices_R12
    self.test_indices_R12 = test_indices_R12

  def _G_acol(R,k):
    '''
    Compute G columns as the average of k batches of rows from matrix R.
    :param k: number of batches (number of columns of matrix G)
    :type k: int
    :return: G matrix of dimensions n x k, where n is the number of rows of R
    :rtype: numpy.ndarray
    '''
    assert k <= R.shape[0] # deberia ponerse al recibir el input (?)
    # random permutation of R row indices
    prows = np.random.permutation(R.shape[0])
    G = list()
    # compute the mean of the elements in a random batch and use it as column of matrix G 
    for batch in np.array_split(prows,k):
      G.append(np.mean(R[batch,:], axis=0))
    return np.array(G).transpose()

  def _G_kmeans(R,k):
    '''
    Compute matrix G as kmeans centroids of matrix R with k clusters.
    :param R: input matrix
    :type R: numpy.ndarray
    :param k: number of clusters
    :type k: int
    :return: matrix G of dimensions n x k, where n is the number of rows of R
    :rtype: numpy.ndarray
    '''
    km = KMeans(n_clusters=k, n_init = 10).fit_predict(R)
    return np.array([np.mean([R[i] for i in range(len(km)) if km[i] == p], axis = 0) for p in set(km)]).transpose()


  def initialize(self):
    '''
    Initialize the G and H matrices
    '''

    if self.init_method == 'acol':
      '''average columns'''

      self.G1 = NMTF._G_acol(self.R12_train.transpose(),self.K[0])
      self.G2 = NMTF._G_acol(self.R23.transpose(),self.K[1])
      self.G3 = NMTF._G_acol(self.R34.transpose(),self.K[2])
      self.G4 = NMTF._G_acol(self.R24,self.K[3])

    if self.init_method == 'kmeans':

      self.G1 = NMTF._G_kmeans(self.R12_train.transpose(),self.K[0])  # clustering patients based on subtypes
      self.G2 = NMTF._G_kmeans(self.R23.transpose(),self.K[1])        # clustering genes based on patients
      self.G3 = NMTF._G_kmeans(self.R34.transpose(),self.K[2])  # clustering miRNAs based on genes
      self.G4 = NMTF._G_kmeans(self.R24,self.K[3])                    # clustering patients based on miRNA

    if self.init_method == 'SVD':
      '''SVD: Singular Value Decomposition'''

      n1,n2 = self.R12_train.shape
      n3,n4 = self.R34.shape

      G1,_,_ = np.linalg.svd(self.R12_train, full_matrices=False)
      G2,_,_ = np.linalg.svd(self.R23, full_matrices=False)
      _,_,G4t = np.linalg.svd(self.R24, full_matrices=False)
      G3,_,_ = np.linalg.svd(self.R34, full_matrices=False)

      self.G1 = np.maximum(np.zeros((n1,self.K[0])) + 1e-5, G1[:,:self.K[0]])    # take only positive entries from the first k1 columns
      self.G2 = np.maximum(np.zeros((n2,self.K[1])) + 1e-5, G2[:,:self.K[1]])    # take only positive entries from the first k2 columns
      self.G3 = np.maximum(np.zeros((n3,self.K[2])) + 1e-5, G3[:,:self.K[2]])    # take only positive entries from the first k3 columns
      self.G4 = np.maximum(np.zeros((n4,self.K[3])) + 1e-5, G4t.T[:,:self.K[3]]) # take only positive entries from the first k3 columns

    # initialize H matrices
    self.H12 = self.G1.transpose() @ self.R12_train @ self.G2  # solving for H12 in Eq (1), assuming Cons (1) and (2)
    self.H23 = self.G2.transpose() @ self.R23 @ self.G3        # solving for H23 in Eq (2), assuming Cons (2) and (3)
    self.H24 = self.G2.transpose() @ self.R24 @ self.G4        # solving for H24 in Eq (3), assuming Cons (2) and (4)
    self.H34 = self.G3.transpose() @ self.R34 @ self.G4  # solving for H34 in Eq (4), assuming Cons (3) and (4)

    self.iter = 0


  def _update(A,num,den):
    '''
    Update rule for the NMTF algorithm
    :param num: numerator of the update rule
    :param den: denominator of the update rule
    '''
    return A * (num/ (den + 1e-8))**0.5

  def iterate(self):
    '''
    Perform one iteration of the NMTF algorithm

    Eq (1). R12 = G1 H12 G2t
    Eq (2). R23 = G2 H23 G3t
    Eq (3). R24 = G2 H24 G4t
    Eq (4). R34 = G3 h34 G4t

    Cons (1). G1t G1 = I
    Cons (2). G2t G2 = I
    Cons (3). G3t G3 = I
    Cons (4). G4t G4 = I
    '''
    
    G1G1t = self.G1 @ self.G1.transpose()
    G2G2t = self.G2 @ self.G2.transpose()
    G3G3t = self.G3 @ self.G3.transpose()
    G4G4t = self.G4 @ self.G4.transpose()

    G1tG1 = self.G1.transpose() @ self.G1 
    G2tG2 = self.G2.transpose() @ self.G2 
    G3tG3 = self.G3.transpose() @ self.G3
    G4tG4 = self.G4.transpose() @ self.G4 

    R12G2H12t = self.R12_train @ self.G2 @ self.H12.transpose() # solving for G1 in Eq (1), assuming Cons (2)

    R12tG1H12 = self.R12_train.transpose() @ self.G1 @ self.H12 # solving for G2 in Eq (1), assuming Cons (2)
    R23G3H23t = self.R23 @ self.G3 @ self.H23.transpose()       # solving for G2 in Eq (2), assuming Cons (3)
    R24G4H24t = self.R24 @ self.G4 @ self.H24.transpose()       # solving for G2 in Eq (3), assuming Cons (4)

    R23tG2H23 = self.R23.transpose() @ self.G2 @ self.H23       # solving for G3 in Eq (2), assuming Cons (3)
    R34G4H34t = self.R34 @ self.G4 @ self.H34.transpose() # solving for G3 in Eq (4), assuming Cons (4)

    R24tG2H24 = self.R24.transpose() @ self.G2 @ self.H24       # solving for G4 in Eq (3), assuming Cons (4)
    R34tG3H34 = self.R34.transpose() @ self.G3 @ self.H34 # solving for G4 in Eq (4), assuming Cons (4)

    G1tR12G2 = self.G1.transpose() @ self.R12_train @ self.G2   # solving for H12 in Eq (1), assuming Cons (1) and (2)
    G2tR23G3 = self.G2.transpose() @ self.R23 @ self.G3         # solving for H23 in Eq (2), assuming Cons (2) and (3)
    G2tR24G4 = self.G2.transpose() @ self.R24 @ self.G4         # solving for H24 in Eq (3), assuming Cons (2) and (4)
    G3tR34G4 = self.G3.transpose() @ self.R34 @ self.G4   # solving for H34 in Eq (4), assuming Cons (3) and (4)

    self.G1 = NMTF._update(self.G1, R12G2H12t, G1G1t@R12G2H12t)
    self.G2 = NMTF._update(self.G2, R12tG1H12 + R23G3H23t + R24G4H24t, G2G2t@R12tG1H12 + G2G2t@R23G3H23t + G2G2t@R24G4H24t)
    self.G3 = NMTF._update(self.G3, R23tG2H23 + R34G4H34t, G3G3t@R23tG2H23 + G3G3t@R34G4H34t)
    self.G4 = NMTF._update(self.G4, R24tG2H24 + R34tG3H34, G4G4t@R24tG2H24 + G4G4t@R34tG3H34)

    self.H12 = NMTF._update(self.H12, G1tR12G2, G1tG1@self.H12@G2tG2)
    self.H23 = NMTF._update(self.H23, G2tR23G3, G2tG2@self.H23@G3tG3)
    self.H24 = NMTF._update(self.H24, G2tR24G4, G2tG2@self.H24@G4tG4)
    self.H34 = NMTF._update(self.H34, G3tR34G4, G3tG3@self.H34@G4tG4)

    self.iter += 1

  def validate(self, metric='aps'):
    '''
    Validate the NMTF algorithm using the AUROC or APS metric
    :param metric: evaluation metric
    :type metric: str
    '''

    R12_found = self.G1 @ self.H12 @ self.G2.transpose()

    # validate predictions only on the training set
    R12_true = self.R12[:,self.train_indices_R12].flatten()
    R12_pred = R12_found[:,self.train_indices_R12].flatten()

    if metric == 'auroc':
      fpr_R12, tpr_R12, threshold_R12 = metrics.roc_curve(R12_true,R12_pred)
      ans_R12 = metrics.auc(fpr_R12, tpr_R12)
    
    elif metric == 'aps':
      ans_R12 = metrics.average_precision_score(R12_true,R12_pred)

    return ans_R12

  def loss(self):
    '''
    Compute the loss function for the NMTF algorithm
    :return: loss function value
    :rtype: float
    '''

    J = np.linalg.norm(self.R12_train - self.G1 @ self.H12 @ self.G2.transpose(), ord='fro')**2
    J += np.linalg.norm(self.R23 - self.G2 @ self.H23 @ self.G3.transpose(), ord='fro')**2
    J += np.linalg.norm(self.R24 - self.G2 @ self.H24 @ self.G4.transpose(), ord='fro')**2
    J += np.linalg.norm(self.R34 - self.G3 @ self.H34 @ self.G4.transpose(), ord='fro')**2
    J += np.trace(self.G3.transpose() @ self.L3 @ self.G3)
    J += np.trace(self.G4.transpose() @ self.L4 @ self.G4)

    return J

