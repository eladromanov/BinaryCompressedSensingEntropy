csv_file_name = 'CSOnly_Results.txt'
# csv_file_name = 'Hadamard.txt'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as stat_norm

labels_dict = {
    'Gaussian+AMP' : 'Dense Gaussian; AMP',
    'LDPC+Glauber_Init0' : 'Sparse LDPC; Glauber (0 init)',
    'LDPC+NNLS' : 'Sparse LDPC; NNLS',
    'LDPC+Glauber_InitNNLS' : 'Sparse LDPC; Glauber, NNLS init',
    'Narayanan+NNLS' : 'Amalladinne et al.\n (BCH matrix; NNLS)'
    }

df = pd.read_csv(csv_file_name, usecols=["Algorithm","K","EbN0db","error"], sep=' ')
df.round({'EbN0db' : 1})

for k, k_grp in df.groupby(["K"]):
    plt.clf() 
    plot_title = "BER vs. Eb/N0 (db), k={0}".format(int(k))
    plt.title(plot_title)
    plt.xlabel("Eb/N0 (db)")
    plt.ylabel("bit error rate")
    fig_file = 'CSOnly_BERVsEbN0, k={0}.pdf'.format(k)
    plt.axhline(y=0.05, color='g', linestyle='--')
    for alg, alg_grp in k_grp.groupby(["Algorithm"]):
        Avg = alg_grp.groupby("EbN0db", as_index=False).mean()
        # Plot and error vs EbN0db curve for this k    
        Array = Avg[["EbN0db","error"]].to_numpy()
        EbN0db = Array[:,0]
        error = Array[:,1]
        
        Scatter = alg_grp[["EbN0db","error"]].to_numpy()
        if alg in labels_dict.keys():
            label = labels_dict[alg]
            plt.plot(EbN0db, error, marker='o', label=label)
            # if alg == 'Narayanan+NNLS':
            #     plt.scatter( Scatter[:,0], Scatter[:,1], s=4)
            #     print(Scatter.shape)                
            
            
            
    plt.legend(fontsize='small',loc='upper right')
    plt.savefig(fig_file)