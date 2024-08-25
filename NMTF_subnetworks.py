import numpy as np
import pandas as pd
import glob
import os
import sys
import sklearn.metrics as skmetrics
import networkx as nx


#-----------------------------------------------------------------------------------
# read input parameters
#-----------------------------------------------------------------------------------

#subtype = str(sys.argv[1]) # 'LumA', 'Her2', 'LumB', 'Basal', 'Normal'
test = True if str(sys.argv[1])=='test' else 'top' if str(sys.argv[4])=='top' else False # all, test

#-----------------------------------------------------------------------------------
# define input folder path
#-----------------------------------------------------------------------------------

if test:
    output_path = '../output_files/test_output_data_1/'
else:
    output_path = '../output_files/output_data/'

#-----------------------------------------------------------------------------------
# read K parameters
#-----------------------------------------------------------------------------------

files_list = glob.glob(output_path+f'Results_attempt_1/init_*') # get all files in the folder (same in all attempts)

init_list = list()
for file in files_list:
    init_list.append(file.split('/')[-1])

init_list.sort()
init_list

K = dict()

for file in init_list:
    prefix,kdims = file.split('_K_')
    method = prefix.split('_')[-1]
    K[method] = kdims

#-----------------------------------------------------------------------------------
# read performance file
#-----------------------------------------------------------------------------------

df_allperformance = pd.read_csv(output_path + 'performance_all_attempts_all_inits_all_runs.csv',index_col=0)

#-----------------------------------------------------------------------------------
# froze on best performance
#-----------------------------------------------------------------------------------

print('Finding best performance combination...',flush=True)

best_aps_R12 = df_allperformance.loc[df_allperformance['aps_R12'].idxmax()]
# best_aps_R12.to_frame().T
attempt = best_aps_R12.attempt
run = best_aps_R12.run
init = best_aps_R12.init
best_path = output_path + 'Results_attempt_{}/init_{}_K_{}/run_{}'.format(attempt,init,K[init],int(run))

Rs = glob.glob(best_path+'/*')

for i in range(len(Rs)):
    which_R = Rs[i].split('/')[-1][:3]
    if which_R == 'R12':
        R12_path = Rs[i]
    elif which_R == 'R23':
        R23_path = Rs[i]
    elif which_R == 'R24':
        R24_path = Rs[i]
    elif which_R == 'R34':
        R34_path = Rs[i] 
        
R12_data = np.load(R12_path,allow_pickle=True)
R34_data = np.load(R34_path,allow_pickle=True)
R23_data = np.load(R23_path,allow_pickle=True)
R24_data = np.load(R24_path,allow_pickle=True)

R12_pred_best = pd.DataFrame(R12_data['values'])
R34_pred_best = pd.DataFrame(R34_data['values']).abs()
R23_pred_best = pd.DataFrame(R23_data['values'])
R24_pred_best = pd.DataFrame(R24_data['values'])

#-----------------------------------------------------------------------------------
# Load original R matrices 
#-----------------------------------------------------------------------------------

print('Loading original R matrices... ',flush=True)

R12_path = '../input_data/Network_matrices/R12.npz'
R23_path = '../input_data/Network_matrices/R23_norm.npz'
R24_path = '../input_data/Network_matrices/R24_norm.npz'
R34_path = '../input_data/Network_matrices/R34_corr.npz'

R12_data = np.load(R12_path,allow_pickle=True)
R23_data = np.load(R23_path,allow_pickle=True)
R24_data = np.load(R24_path,allow_pickle=True)
R34_data = np.load(R34_path,allow_pickle=True)

R12_true_df = pd.DataFrame(R12_data['values'],index=R12_data['ix'],columns=R12_data['cols'])
R23_true_df = pd.DataFrame(R23_data['values'],index=R23_data['ix'],columns=R23_data['cols'])
R24_true_df = pd.DataFrame(R24_data['values'],index=R24_data['ix'],columns=R24_data['cols'])
R34_true_df = pd.DataFrame(R34_data['values'],index=R34_data['ix'],columns=R34_data['cols']).abs()

#-----------------------------------------------------------------------------------
# align indices and columns 
#-----------------------------------------------------------------------------------

print('Aligning columns and indices...',flush=True)

# check rows of R23 and R24 match in the same order with columns of R12
if not (all(R12_true_df.columns == R23_true_df.index) and all(R12_true_df.columns == R24_true_df.index)):
    print('Rows of R23 and R24 does not match in the same order with columns of R12')
# check rows of R34 match in the same order with columns of R23
if not all(R23_true_df.columns == R34_true_df.index):
    print('Rows of R34 does not match in the same order with columns of R23')
# check columns of R34 match in the same order with columns of R24
if not all(R24_true_df.columns == R34_true_df.columns):
    print('Columns of R34 does not match in the same order with columns of R24')

if test == True:
    n1,n2 = R12_true_df.shape
    n3,n4 = 100,80

    R12_true_df = R12_true_df.iloc[:,:n2]
    R23_true_df = R23_true_df.iloc[:n2,:n3]
    R24_true_df = R24_true_df.iloc[:n2,:n4]
    R34_true_df = R34_true_df.iloc[:n3,:n4]

R12_pred_best.columns = R12_true_df.columns
R12_pred_best.index = R12_true_df.index
R23_pred_best.columns = R23_true_df.columns
R23_pred_best.index = R23_true_df.index
R24_pred_best.columns = R24_true_df.columns
R24_pred_best.index = R24_true_df.index
R34_pred_best.columns = R34_true_df.columns
R34_pred_best.index = R34_true_df.index

#-----------------------------------------------------------------------------------
# load co-expression matrices
#-----------------------------------------------------------------------------------

W4_path = '../input_data/Network_matrices/W24_bin.npz'
W3_path = '../input_data/Network_matrices/W23_bin.npz'

W3_data = np.load(W3_path,allow_pickle=True)
W4_data = np.load(W4_path,allow_pickle=True)

W3_df = pd.DataFrame(W3_data['values'],index=W3_data['ix'],columns=W3_data['ix'])
W4_df = pd.DataFrame(W4_data['values'],index=W4_data['ix'],columns=W4_data['ix'])

W4_df = W4_df[list(R24_true_df.columns)]            #change columns order
W4_df = W4_df.reindex(list(R24_true_df.columns))    #change rows order
W3_df = W3_df[list(R23_true_df.columns)]
W3_df = W3_df.reindex(list(R23_true_df.columns))

#-----------------------------------------------------------------------------------
# Binarize R12 
#-----------------------------------------------------------------------------------

print('Binarize R12 predictions...',flush=True)

labels_argmax = np.argmax(R12_pred_best.values,axis=0)
R12_argmax = np.zeros(R12_pred_best.shape)
cols = np.array(range(R12_pred_best.shape[1]))
R12_argmax[labels_argmax,cols] = 1
R12_pred_bin = pd.DataFrame(R12_argmax,columns=R12_pred_best.columns,index=R12_pred_best.index)

#-----------------------------------------------------------------------------------
# Compute NMTF prediction metrics
#-----------------------------------------------------------------------------------
#df_metrics = pd.DataFrame(columns=['attempt','subtype','TN','FP','FN','TP'])
#subtype_list = ['LumA','LumB','Basal','Her2','Normal']

#for subtype in subtype_list:
#    tn, fp, fn, tp = skmetrics.confusion_matrix(R12_true_df.loc[subtype,:].values,R12_pred_bin.loc[subtype,:].values).ravel()
#    df_metrics = df_metrics._append({'attempt':attempt,'subtype':subtype,'TN':tn,'FP':fp,'FN':fn,'TP':tp},ignore_index=True)

#df_metrics.to_csv(output_path+'subtype_metrics_by_attempt.csv',index=True,header=True)

#-----------------------------------------------------------------------------------
# create combined adjacency matrix and corresponding graph
#-----------------------------------------------------------------------------------

print('Creating combinated network...', flush=True)

node_names = R23_pred_best.index.tolist() + R23_pred_best.columns.tolist() + R24_pred_best.columns.tolist()
# node_names = R12.index.tolist() + R23.index.tolist() + R23.columns.tolist() + R24.columns.tolist()
A = pd.DataFrame(columns = node_names,index = node_names)

# A.loc[R12.index,R12.columns] = R12_bin.values
# A.loc[R12.index,R12.columns] = R12.values
A.loc[R23_pred_best.index,R23_pred_best.columns] = R23_pred_best.values
A.loc[R24_pred_best.index,R24_pred_best.columns] = R24_pred_best.values
A.loc[R34_pred_best.index,R34_pred_best.columns] = R34_pred_best.values
# A.loc[R12.columns,R12.index] = R12_bin.T.values
# A.loc[R12.columns,R12.index] = R12.T.values
A.loc[R34_pred_best.columns,R34_pred_best.index] = R34_pred_best.T.values
A.loc[R24_pred_best.columns,R24_pred_best.index] = R24_pred_best.T.values
A.loc[R23_pred_best.columns,R23_pred_best.index] = R23_pred_best.T.values
A.loc[R23_pred_best.columns,R23_pred_best.columns] = W3_df.values
A.loc[R24_pred_best.columns,R24_pred_best.columns] = W4_df.values
A.fillna(0,inplace=True)

#-----------------------------------------------------------------------------------
# Create network
#-----------------------------------------------------------------------------------

# node names to numbers
#A.columns = range(A.shape[1])
#A.index = range(A.shape[0])

G = nx.from_pandas_adjacency(A)
print(G,flush=True)

#-----------------------------------------------------------------------------------
#Create subnetwork and save edgelist for each subtype 
#-----------------------------------------------------------------------------------

# create folder 'Edgelists' inside output_path if it does not exist
if not os.path.exists(output_path+'Edgelists'):
    os.makedirs(output_path+'Edgelists')

subtype_list = R12_true_df.index.tolist()

for subtype in subtype_list:
    G_sub = G.copy()
    nodes_to_remove = R12_pred_bin.T[R12_pred_bin.T[subtype]==1].index.tolist()
    G_sub.remove_nodes_from(nodes_to_remove)
    print(f'Network {subtype}',flush=True)
    print(G_sub,flush=True)

    #Create edgelist and save
    df = nx.to_pandas_edgelist(G_sub)
    df.to_csv(output_path + 'Edgelists/G_sub_best_aps_{}.edge'.format(subtype),sep='\t',header=False,index=False)


print('Done.',flush=True)
