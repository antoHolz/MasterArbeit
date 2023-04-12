import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore") 
font = {'size'   : 7}

matplotlib.rc('font', **font)

csvdir='C:/Users/Usuario/Documents/Masterarbeit/code/gym-pybullet-drones-1.0.0/files/csvs/Experiments_v3/ATT/'
dirnames=['SO_40_const_eps0.3_ls.10.15',"SO_40_const_eps0.3_ls.15.15","SO_40_const_eps0.4_ls.15.15"]#['baseline', 'ET_10', 'SO_10']
c=['purple','green', 'red', 'orange', 'black','blue']
fl=['thistle','lightgreen', 'lightcoral', 'moccasin', 'lightgray','lightblue']
m=['o','x','v','s','p','^']
sufix='/candidate_performance.csv'
sufix2='/learn_rounds.csv'
legend_GP=['_nolegend_']*(len(dirnames)*2)
legend_GP[::2]=['eps.3_ls.1.15','eps.3_ls.15.15','eps.4_ls.15.15']#['baseline','SafeOpt\u221E','TVSafeOpt10']

f, ax = plt.subplots(2, 1, figsize=(6, 4))
ax[0].axvline(x = 20, color = 'k', linestyle='dashed', label = '_nolegend_')
#ax[0].axvline(x = 39, color = 'k', linestyle='dashed', label = '_nolegend_')
for index, dirname in enumerate(dirnames):
    dirs=[f.path for f in os.scandir(csvdir+dirname) if f.is_dir()]
    performances=pd.DataFrame()
    for i in range(len(dirs)):
        p=pd.read_csv(dirs[i]+sufix)['cost'].dropna(axis=0)
        p=p.reset_index(drop=True)
        performances['cost'+str(i)]=p
    #print(performances.head())
    p_mean=performances.mean(axis=1)
    p_std=performances.std(axis=1)
    ax[0].plot(-p_mean, color=c[index])
    ax[0].fill_between(performances.index.to_numpy(), -p_mean-p_std, -p_mean+p_std, alpha=0.5,color=fl[index])
    ax[0].set_xlim(0, 89)#len(p.index))
ax[0].legend(legend_GP, loc=0)



for index, dirname in enumerate(dirnames):
    dirs=[f.path for f in os.scandir(csvdir+dirname) if f.is_dir()]
    ts=pd.DataFrame()
    for i in range(len(dirs)):
        try:
            ts=pd.concat([ts,pd.read_csv(dirs[i]+sufix2)['timestep'].dropna(axis=0)])
            ts=ts.reset_index(drop=True)
        except FileNotFoundError:
            # print(i)
            continue;
    print(dirname)
    print(ts.value_counts())
    
    l_points=[]
    for i in range(len(dirs)):
        if(os.path.exists(dirs[i]+sufix2)):
            for ts in pd.read_csv(dirs[i]+sufix2)['timestep'].dropna(axis=0):
                l_points.append((ts+index*0.1,i))
    if l_points:
        l_points=np.array(l_points)
        ax[1].scatter(l_points[:,0], l_points[:,1], marker=m[index], s=10, color=c[index], alpha=.35)
        ax[1].set_xlim(0, 59 )#len(p.index)
        ax[1].legend(['TVSafeOpt10', 'TVsafeOpt\u221E'], loc=2)#(dirnames, loc=2)

plt.setp(ax[0], ylabel='Cost')
plt.setp(ax[1], ylabel='Trigger', xlabel='Rounds')


plt.savefig(csvdir+'figures/comparison_alg_bog.png')
plt.show()





