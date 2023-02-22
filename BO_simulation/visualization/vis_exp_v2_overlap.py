import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore") 

csvdir='C:/Users/Usuario/Documents/Masterarbeit/code/gym-pybullet-drones-1.0.0/files/csvs/'
dirnames=['ETs_10learn_bconst', 'ETs_10learn_bconst_exp1'] #arg

f, ax = plt.subplots(1, 1, figsize=(6, 4))
for dirname in dirnames:
    dirs=[f.path for f in os.scandir(csvdir+dirname) if f.is_dir()]
    sufix='/candidate_performance.csv'

    performances=pd.DataFrame()
    for i in range(len(dirs)):
        p=pd.read_csv(dirs[i]+sufix)['cost'].dropna(axis=0)
        performances['cost'+str(i)]=p
    #print(performances.head())

    p_mean=performances.mean(axis=1)
    p_std=performances.std(axis=1)
    (-p_mean).plot()
    ax.fill_between(performances.index.to_numpy(), -p_mean-p_std, -p_mean+p_std, alpha=0.5, label='_nolegend_') #color="thistle"

plt.legend(dirnames)




#plt.savefig(csvdir+'figures/bog_ET10_vs_SOinfty_exp.png')
plt.show()





