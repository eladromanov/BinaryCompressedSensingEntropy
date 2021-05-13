csv_file_name = 'CS_Exp.txt'
# csv_file_name = 'Hadamard.txt'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as stat_norm



def Prepare_Data(csv_file_name = 'CS_Exp.txt', Make_Plots=True):

    
    # Expected format: Algorithm,K,EbN0db,error
    df = pd.read_csv(csv_file_name, usecols=["Algorithm","K","EbN0db","error"])
    
    Dict_Alg={}
    
    for alg, alg_grp in df.groupby(["Algorithm"]):
        
        Dict = {}
        
        for k, k_grp in alg_grp.groupby(["K"]):
            
            k_grp.round(2)
            Avg = k_grp.groupby("EbN0db", as_index=False).mean()
            
            Array = Avg[["EbN0db","error"]].to_numpy()
            Dict[k] = Array
                        
            if Make_Plots:
            
                # Plot and error vs EbN0db curve for this k    
            
                EbN0db = Array[:,0]
                error = Array[:,1]
                
                plot_title = "Error vs. Eb/N0 (db): alg={0}, k={1}".format(alg,int(k))
                plt.clf()
                plt.title(plot_title)
                plt.xlabel("Eb/N0 (db)")
                plt.ylabel("per-user error probability")
                plt.plot(EbN0db, error, marker='o')
                plt.axhline(y=0.05, color='g', linestyle='--')
                fig_file = 'ErrVsEbN0_alg={0},k={1}.pdf'.format(alg, k)
                
                plt.savefig(fig_file,bbox_inches='tight')
            
        Dict_Alg[alg] = Dict
                
    return Dict_Alg
            

def Error_Per_K(csv_file_name = 'CS_Exp.txt'):
    
    # Expected format: Algorithm,K,EbN0db,error
    df = pd.read_csv(csv_file_name, usecols=["Algorithm","K","EbN0db","error"])
              
    for k, k_Grp in df.groupby(["K"]):
        
        k_Grp.round(2)
        
        plt.clf()
        plt.title("Error vs Eb/N0 at K={0}".format(k) )
        plt.xlabel("Eb/N0 (db)")
        plt.ylabel("per-user error probability")
        
        for Alg, Alg_Grp, in k_Grp.groupby(["Algorithm"]):
            
            Avg = Alg_Grp.groupby("EbN0db", as_index=False).mean()
            Array = Avg[["EbN0db","error"]].to_numpy()
            print(Array)
            plt.plot(Array[:,0], Array[:,1], label=Alg, marker='o')
        
        fig_file = 'ErrProbComparison_k={0}.pdf'.format(k)
        plt.legend()
        plt.axhline(y=0.05, color='g', linestyle='--')
        plt.savefig(fig_file,bbox_inches='tight')
            
            


"""
Returns the required EbN0 to achieve the given parameters in AWGN transmission.
Done using the channel dispersion bound of Polyanskiy:
    payload/n ~= C - sqrt{V/n} * invQ(error)

CAUTION: Returned EbN0 is NOT in dbs.

"""
def AWGN_Error_To_EbN0(payload, n, error):
    
    Rate = payload/n
    
    def F(P):
        C = 1/2 * np.log2(1+P)
        V = P*(P+2)/(P+1)**2 * (1/2) * (np.log2(np.e))**2
        return C - np.sqrt(V/n) * stat_norm.isf(error)
    
    # Solve for P: F(P) = Rate. Since F(P) is increasing, do by binary search
    P_low = 0
    P_high = 1
    while F(P_high) < Rate:
        P_low = P_high
        P_high = 2*P_high
    
    P = (P_low+P_high)/2
    PREC=0.000001
    while (P_high-P_low)>PREC:
        P = (P_low+P_high)/2
        if F(P)<Rate:
            P_low=P
        else:
            P_high=P
            
    # Calculate corresponding EbN0 from P.
    # Total energy: E = n*P = E_b*payload ,   where E_b=2*EbN0 (N0=2sigma^2=2) is energy per bit.
    EbN0 = (1/2) * (n*P)/payload
    
    return EbN0
    
    
    
# print( AGWN_Error_To_EbN0(100, 50, 0.01) )

Data_Alg = Prepare_Data(csv_file_name=csv_file_name)

# Simulation parameters
n = 30000   # channel uses
n1 = 2**11  # channel users for stage 1
B = 100     # total per-user payload
J = 14      # bits for first stage
target_err = 0.05   # target per-user error probability
AWGN_payload = B-J  # per-user payload remaning for AWGN stage

labels_dict = {
        'Glauber+NNLS' : 'Proposed: sparse LDPC; Glauber, NNLS init',
        'AMP' : 'AMP (dense Gaussian i.i.d.)'
    }

plt.clf()    
for (alg, DataDict) in Data_Alg.items():

    Ks = []
    TradeOff = []  # How much EbN0 (db) to put on CS step
    EbN0_db = []  # How much  EbN0 (db) one needs for end-to-end transmission


    for (k, Data) in DataDict.items():
        
        Ks.append(k)
        n2 = int( (n-n1)/k )  # AWGN channel uses per user, where he get a slot without interference
        
        curveEbN0 = 0
        Achieved_EbN0db = float('inf')
        
        for i in range(Data.shape[0]):
            
            curveError = Data[i,1]
            curveEbN0db = Data[i,0]
            curveEbN0 = 10**(curveEbN0db/10)
            
            # print(curveEbN0db,curveEbN0,curveError)
            
            if curveError < target_err:
                
                awgnReqEbN0 = AWGN_Error_To_EbN0(AWGN_payload, n2, target_err - curveError)
                
                totEbN0 = (J*curveEbN0 + AWGN_payload*awgnReqEbN0)/B
                totEbN0db = 10 * np.log10(awgnReqEbN0)
                # print(totEbN0db)
                if totEbN0db < Achieved_EbN0db:
                    Achieved_EbN0db = totEbN0db
                    Best_TradeOff = curveEbN0db
                    
        # sanityCheckEbN0 = AWGN_Error_To_EbN0(AWGN_payload, n2, 0.001)
        # sanityCheckEbN0db = 10*np.log10(sanityCheckEbN0)
        # print('sanity check: ', k, sanityCheckEbN0db )    
        
        TradeOff.append(Best_TradeOff)
        EbN0_db.append(Achieved_EbN0db)
        # print('---')
    
    # plt.clf()    
    # print(alg, TradeOff)
    
    label = labels_dict[alg]
    
    plt.plot(Ks, EbN0_db, marker='o', label=label)
    # plt.title("End-to-end required Eb/N0 vs k, ALG={0}".format(alg))
    # plt.xlabel("k")
    # plt.ylabel("Eb/N0 (db)")
    plt.xticks(ticks=Ks)
    # plt.savefig("EndToEnd_{0}.pdf".format(alg))
    
    
plt.title("End-to-end required Eb/N0 vs k")    
plt.xlabel("k")
plt.ylabel("Eb/N0 (db)")    


# Data for Facenda-Silva.
ks = [50,100,150,200,250,300]
silva_data = [1.6,2.1,2.9,3.1,4.2,5.4]
plt.plot(ks,silva_data, label='Facenda-Silva [12]',marker='o')

plt.legend()
plt.savefig('EndToEnd_All.pdf')












# Error_Per_K(csv_file_name)

    
    
    
    
    
    
    
    
    
    
    
    
    
    