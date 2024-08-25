import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class mask_generator():
	'''
    Class to generate the training and testing masks for R12.
	'''

	def __init__(self,R12_df=None,sample_frac_R12=None):

		self.R12_df = R12_df 
		self.sample_frac_R12 = sample_frac_R12
		self.train_rows_ix_R34 = None
		self.train_cols_ix_R34 = None

	def get_MatrixTrain_and_Masks_R12(self):
		'''
        Generates training and testing masks for the R12 dataset.
    
        The columns of the dataset, which represent different patients, are divided into masked and unmasked columns. 
        The unmasked columns are used as the training set, while the masked columns are divided into validation and testing sets.
        
        Masked columns are assigned a value of 0.5, whereas unmasked columns remain unchanged. The resulting training set consists of unmasked columns.
        The function supports both manual and stratified sampling methods.
        
        Sampling Methods:
        - **Manual Sampling**: If `sample_frac_R12` is `None`, manual sampling is applied where 'Normal' class samples are capped at 20, and other classes at 40.
        - **Stratified Sampling**: If `sample_frac_R12` is provided, the sampling is performed such that the final sample retains the same class proportions as the initial dataset.
        
		The masked columns are further split into 50/50 validation and testing sets.

        :returns: 
            - `matrix_train_R12`: A NumPy array with masked columns filled with 0.5 for training purposes.
            - `mask_train_test`: A DataFrame with columns 'column_name', 'class', and 'group' indicating the assignment of each column to the training, testing or validation set.
            
        :rtype: (np.ndarray, pd.DataFrame)
		'''
		
		self.matrix_R12 = np.copy(self.R12_df.values)

		self.R12_df[self.R12_df.values == 0] = np.nan
		mask_train_test = self.R12_df.T.stack().reset_index()
		mask_train_test.drop(0,inplace=True,axis=1)
		mask_train_test.columns = ['column_name','class']

		if self.sample_frac_R12 is None: # apply manual sampling method 
			sample_df = pd.DataFrame()
			for c in mask_train_test['class'].unique():
				min_sample = len(mask_train_test[mask_train_test['class'] == c])
				if c == 'Normal':
					subsample = mask_train_test[mask_train_test['class'] == c].sample(n = min(20,min_sample))
				else:
					subsample = mask_train_test[mask_train_test['class'] == c].sample(n = min(40,min_sample))
				sample_df = pd.concat([sample_df,subsample])

		else: # stratified sampling where the final sample has the same proportion of classes as the initial population
			sample_df = mask_train_test.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=self.sample_frac_R12))
		
        # Spliting the sample in training and testing 50/50
		training_df = sample_df.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=0.5))
	    
		self.mask_ix_R12 = sample_df.index.tolist()
		self.training_ix_R12 = training_df.index.tolist()
		self.testing_ix_R12 = list(set(self.mask_ix_R12).difference(self.training_ix_R12))

		mask_train_test['group'] = ['training' if i in self.training_ix_R12 \
		                        else 'testing' if i in self.testing_ix_R12 \
		                        else 'validation' for i in range(mask_train_test.shape[0])]

		self.matrix_train_R12 = np.copy(self.matrix_R12)

		for j in self.mask_ix_R12:
			self.matrix_train_R12[:,j] = 0.5

        	
		return self.matrix_train_R12, mask_train_test