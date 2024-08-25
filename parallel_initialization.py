import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

from mask_train_test import mask_generator
from method_NMTF import NMTF


print('Starting time: ',datetime.datetime.now(),flush=True)

test = True if str(sys.argv[1])=='test' else False # True or False

n_jobs = -1

if test:
    base_output_path = '../output_files/test_output_data/'
else:
    base_output_path = '../output_files/output_data/'

# Start with the base output path
output_path = base_output_path

# Initialize a counter for differentiating the directory name
counter = 1

# Check if the directory exists, and if it does, append a counter until we find a unique name
while os.path.exists(output_path):
    output_path = base_output_path[:-1] + f"_{counter}/"
    counter += 1

# Create the directory
os.makedirs(output_path)
print(f"Directory '{output_path}' created.")

R12_path = '../input_data/Network_matrices/R12.npz'
R23_path = '../input_data/Network_matrices/R23_norm.npz'
R24_path = '../input_data/Network_matrices/R24_norm.npz'
R34_path = '../input_data/Network_matrices/R34_corr.npz'
W3_path = '../input_data/Network_matrices/W23_bin.npz'
W4_path = '../input_data/Network_matrices/W24_bin.npz'

total_attempts = 10
total_runs = 3
max_iter = 5000
rule_of_thumb = True
stratified_sample_R12 = True

val_metric = 'aps' # validation metric ('aps': average presicion, 'auroc':Area Under the Receiver Operating Characteristics)

init_methods = ['kmeans','SVD']

R12_data = np.load(R12_path,allow_pickle=True)
R23_data = np.load(R23_path,allow_pickle=True)
R24_data = np.load(R24_path,allow_pickle=True)
R34_data = np.load(R34_path,allow_pickle=True)
W3_data = np.load(W3_path,allow_pickle=True)
W4_data = np.load(W4_path,allow_pickle=True)

R12_true_df = pd.DataFrame(R12_data['values'],index=R12_data['ix'],columns=R12_data['cols'])
R23_df = pd.DataFrame(R23_data['values'],index=R23_data['ix'],columns=R23_data['cols'])
R24_df = pd.DataFrame(R24_data['values'],index=R24_data['ix'],columns=R24_data['cols'])
R34_df = pd.DataFrame(R34_data['values'],index=R34_data['ix'],columns=R34_data['cols']).abs()
W3_df = pd.DataFrame(W3_data['values'],index=W3_data['ix'],columns=W3_data['ix'])
W4_df = pd.DataFrame(W4_data['values'],index=W4_data['ix'],columns=W4_data['ix'])

W4_df = W4_df[list(R24_df.columns)]           #change columns order
W4_df = W4_df.reindex(list(R24_df.columns))   #change rows order
W3_df = W3_df[list(R23_df.columns)]
W3_df = W3_df.reindex(list(R23_df.columns))

R12 = R12_true_df.values
R34 = R34_df.values
R23 = R23_df.values
R24 = R24_df.values

if test == True:
    n1,n2 = R12.shape
    n3,n4 = 100,80

    R12 = R12[:,:n2]
    R23 = R23[:n2,:n3]
    R24 = R24[:n2,:n4]
    R34 = R34[:n3,:n4]

    R12_true_df = R12_true_df.iloc[:,:n2]
    R23_df = R23_df.iloc[:n2,:n3]
    R24_df = R24_df.iloc[:n2,:n4]
    R34_df = R34_df.iloc[:n3,:n4]
    W3_df = W3_df.iloc[:n3,:n3]
    W4_df = W4_df.iloc[:n4,:n4]

L3 = np.diag(W3_df.values.sum(0)) - W3_df.values
L4 = np.diag(W4_df.values.sum(0)) - W4_df.values

n1,n2 = R12.shape
n3,n4 = R34.shape

n_dims = [n1,n2,n3,n4]

nodes = [n2,n3,n4,n2] # clustering patients(n2), genes(n3), miRNAs(n4), patients(n2)
ad_hoc_dims = [5,5,5,6]

# Define dimensionality of factor matrices
K = dict()

if rule_of_thumb:
    def rot(n):
        return round((n/2)**0.5)
    k_vals = [max(rot(nodes[i]),ad_hoc_dims[i]) for i in range(len(nodes))]
    for i in range(len(init_methods)):
        K[init_methods[i]] = k_vals.copy()

else:
    for i in range(len(init_methods)):
        K[init_methods[i]] = ad_hoc_dims


if 'SVD' in K.keys():
    K['SVD'][0] = 5


with open(output_path + 'parameters.txt','w') as f:
    f.write('Total attempts (training set sampling) : ' + str(total_attempts)+'\n')
    if stratified_sample_R12: sampling_strategy_R12 = 'stratified'
    else: sampling_strategy_R12 = 'ad hoc'
    f.write('Sampling strategy R12: ' + sampling_strategy_R12 + '\n')
    f.write('Total runs per init : ' + str(total_runs)+'\n')
    f.write('Maximum number of iterations per run : ' + str(max_iter)+'\n')
    f.write('validation metric : ' + str(val_metric)+'\n')
    if rule_of_thumb: clusters_dim_estimation = 'rule of thumb'
    else:  clusters_dim_estimation = 'ad hoc'
    f.write('Clusters dimensions : '+ clusters_dim_estimation +'\n')
    for init in init_methods:
        k_text = ','.join(['k'+str(i+1) for i in range(len(K[init]))])
        k_vals = ','.join([str(k) for k in K[init]])
        f.write('For {0} initialization: ({1}) = {2}'.format(init,k_text,k_vals)+'\n')
        for j in range(len(nodes)):
            assert K[init][j] <= nodes[j]


def attempt_loop(attempt):
    print('Start attempt {0}/{1} : {2}'.format(attempt,total_attempts,datetime.datetime.now()),flush=True)

    attempt_dir = output_path+'Results_attempt_{0}'.format(attempt) 
    os.mkdir(attempt_dir)

    generator = mask_generator(R12_true_df,sample_frac_R12=0.4)

    R12_training, mask_R12 = generator.get_MatrixTrain_and_Masks_R12()
    mask_R12.to_csv(attempt_dir+'/mask_train_test_R12.csv')
          
    Parallel(n_jobs=n_jobs)(delayed(init_loop)(init,
                                          attempt_dir,
                                          R12_training, R34_df.values,
                                          generator.training_ix_R12,
                                          generator.testing_ix_R12) for init in init_methods)

def init_loop(init,attempt_dir,R12_train,R34,
              training_indices_R12,testing_indices_R12):
    print('Start init method {0} : {1}'.format(init,datetime.datetime.now()),flush=True)

    k_vals = '_'.join([str(k) for k in K[init]])
    init_dir = attempt_dir+'/init_{0}_K_{1}'.format(init,k_vals) 
    os.mkdir(init_dir)

    nmtf = NMTF(R12,R12_train,R23,R24,R34,L3,L4,init,K[init],
                training_indices_R12,testing_indices_R12)
    
    if (init == 'SVD'):
        t_runs = 1 # There is no need to run multiple times for SVD initialization (deterministic)
    else:
        t_runs = total_runs
    result = Parallel(n_jobs=n_jobs)(delayed(run_loop)(run,init_dir,nmtf) for run in range(1,t_runs+1))    

    performance = pd.concat(result,ignore_index=True)
    
    #save performance info
    performance.to_csv(init_dir+'/performance_all_runs.csv')
    
    
def run_loop(run,init_dir,nmtf):
    print('Start run {0} : {1}'.format(run,datetime.datetime.now()),flush=True)
    
    run_dir = init_dir+'/run_{0}'.format(run) 
    os.mkdir(run_dir)

    nmtf.initialize()
    max_val_R12 = -np.infty

    val_R12 = nmtf.validate(val_metric)
    performance = pd.DataFrame(columns=['run','nmtf_iter','loss',val_metric+'_R12','cum_time'])
    performance = performance.append({'run':run, 'nmtf_iter':nmtf.iter, 'loss':nmtf.loss(),
                                       val_metric+'_R12':val_R12,'cum_time':0},
                                       ignore_index=True)
    start_time = time.time()

    while (nmtf.iter < max_iter):

        nmtf.iterate()

        if nmtf.iter % 10 == 0:

            validate_next_R12 = nmtf.validate(val_metric)

            performance = performance.append({'run':run, 'nmtf_iter':nmtf.iter, 'loss':nmtf.loss(),
                                               val_metric+'_R12':validate_next_R12,
                                               'cum_time': time.time() - start_time},
                                               ignore_index=True)

            if validate_next_R12 > max_val_R12:
                max_val_R12 = validate_next_R12
                print(nmtf.iter,' R12 ',validate_next_R12,flush=True)
                iter_max_val_R12 = nmtf.iter
                R12_on_max_val_R12 = nmtf.G1 @ nmtf.H12 @ nmtf.G2.transpose()
                R23_on_max_val_R12 = nmtf.G2 @ nmtf.H23 @ nmtf.G3.transpose()
                R24_on_max_val_R12 = nmtf.G2 @ nmtf.H24 @ nmtf.G4.transpose()
                R34_on_max_val_R12 = nmtf.G3 @ nmtf.H34 @ nmtf.G4.transpose()


    # save R12_on_max_val
    np.savez_compressed(run_dir+'/R12_on_max_{0}_for_R12_iter_{1}.npz'.format(val_metric,iter_max_val_R12),values=R12_on_max_val_R12) 
    np.savez_compressed(run_dir+'/R23_on_max_{0}_for_R12_iter_{1}.npz'.format(val_metric,iter_max_val_R12),values=R23_on_max_val_R12)
    np.savez_compressed(run_dir+'/R24_on_max_{0}_for_R12_iter_{1}.npz'.format(val_metric,iter_max_val_R12),values=R24_on_max_val_R12) 
    np.savez_compressed(run_dir+'/R34_on_max_{0}_for_R12_iter_{1}.npz'.format(val_metric,iter_max_val_R12),values=R34_on_max_val_R12)  

    return performance

t = time.time()

Parallel(n_jobs=n_jobs)(delayed(attempt_loop)(attempt) for attempt in range(1,total_attempts+1))

print('Total time = ',time.time()-t,flush=True)
