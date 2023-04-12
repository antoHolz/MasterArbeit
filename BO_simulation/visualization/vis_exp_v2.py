import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore") 

csvdir='C:/Users/Usuario/Documents/Masterarbeit/code/gym-pybullet-drones-1.0.0/files/csvs/'
dirname='ETs_inftylearn_const_wchecks' #arg
dirs=[f.path for f in os.scandir(csvdir+dirname) if f.is_dir()]
sufix1='/candidate_performance.csv'
sufix2='/learn_rounds.csv'

performances=pd.DataFrame()
for i in range(len(dirs)):
    p=pd.read_csv(dirs[i]+sufix1)['cost'].dropna(axis=0)
    p=p.reset_index(drop=True)
    #print(p.iloc[59])
    performances['cost'+str(i)]=p
#print(performances.head())
l_points=[]
for i in range(len(dirs)):
    if(os.path.exists(dirs[i]+sufix2)):
        for ts in pd.read_csv(dirs[i]+sufix2)['timestep'].dropna(axis=0):
            l_points.append((ts,i))
l_points=np.array(l_points)

f, ax = plt.subplots(2, 1, figsize=(6, 4))
p_mean=performances.mean(axis=1)
p_std=performances.std(axis=1)
ax[0].plot(-p_mean, color='purple')
ax[0].fill_between(performances.index.to_numpy(), -p_mean-p_std, -p_mean+p_std, alpha=0.5,color='thistle')

ax[1].scatter(l_points[:,0], l_points[:,1], marker='o', color='purple')
ax[0].set_xlim(0, len(p.index))
ax[1].set_xlim(0, len(p.index))
plt.savefig(csvdir+dirname+'/mean_cost.png')
plt.show()


ts=pd.DataFrame()
sufix='/learn_rounds.csv'
for i in range(len(dirs)):
    try:
        ts=pd.concat([ts,pd.read_csv(dirs[i]+sufix)['timestep'].dropna(axis=0)])
        ts=ts.reset_index(drop=True)
    except FileNotFoundError:
        print(i)
        continue;

print(ts.value_counts())



