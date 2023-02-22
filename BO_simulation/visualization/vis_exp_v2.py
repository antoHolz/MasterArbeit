import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore") 

csvdir='C:/Users/Usuario/Documents/Masterarbeit/code/gym-pybullet-drones-1.0.0/files/csvs/'
dirname='ETs_10learn_bconst' #arg
dirs=[f.path for f in os.scandir(csvdir+dirname) if f.is_dir()]
sufix='/candidate_performance.csv'

performances=pd.DataFrame()
for i in range(len(dirs)):
    p=pd.read_csv(dirs[i]+sufix)['cost'].dropna(axis=0)
    print(p.iloc[59])
    performances['cost'+str(i)]=p
#print(performances.head())


f, ax = plt.subplots(1, 1, figsize=(6, 4))
p_mean=performances.mean(axis=1)
p_std=performances.std(axis=1)
(-p_mean).plot(color='purple')
ax.fill_between(performances.index.to_numpy(), -p_mean-p_std, -p_mean+p_std, alpha=0.5,color='thistle')

plt.savefig(csvdir+dirname+'/mean_cost.png')
plt.show()


ts=pd.DataFrame()
sufix='/learn_rounds.csv'
for i in range(len(dirs)):
    try:
        ts=pd.concat([ts,pd.read_csv(dirs[i]+sufix)['timestep'].dropna(axis=0)])
    except FileNotFoundError:
        print(i)
        continue;

print(ts.value_counts())



