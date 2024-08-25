import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pecanpy import pecanpy as node2vec

dim = int(sys.argv[1])
num_walks = int(sys.argv[2])
walk_length = int(sys.argv[3])
network = str(sys.argv[4]) # 'LumA', 'Her2', 'LumB', 'Basal', 'Normal'
test = True if str(sys.argv[5])=='test' else False

if test:
    output_path = '../output_files/test_output_data_1/'
else:
    output_path = '../output_files/output_data/'

path_to_edg_file = output_path + 'Edgelists/G_sub_best_aps_{}.edge'.format(network)

# initialize node2vec object, similarly for SparseOTF and DenseOTF
print('initialize node2vec object...',flush=True)
g = node2vec.DenseOTF(p=1, q=1, workers=-1, verbose=True, extend=True)
# alternatively, can specify ``extend=True`` for using node2vec+

# load graph from edgelist file
print('load graph from file...',flush=True)
g.read_edg(path_to_edg_file, weighted=True, directed=False)
# precompute and save 2nd order transition probs (for PreComp only)
print('precompute transition probabilities...',flush=True)
g.preprocess_transition_probs()

# generate random walks, which could then be used to train w2v
# walks = g.simulate_walks(num_walks=10, walk_length=80)

# alternatively, generate the embeddings directly using ``embed``
# emd = g.embed(dim=500)
print('generate embeddings...',flush=True)
emd = g.embed(dim=dim, num_walks=num_walks, walk_length=walk_length)

print('save embeddings...',flush=True)

# create folder 'Embeddings' in the output directory if it does not exist
import os
if not os.path.exists(output_path + 'Embeddings'):
    os.makedirs(output_path + 'Embeddings')

emd_df = pd.DataFrame(emd, index=g.nodes)
emd_df.to_csv(output_path + 'Embeddings/embed_pred_DenseOTF_extend_{}_{}_{}_{}.emb'.format(dim,num_walks,walk_length,network))

print('Done.',flush=True)
