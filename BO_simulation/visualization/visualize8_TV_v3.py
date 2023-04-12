import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os


R=.3
NUM_WP=48*6
NUM_R=48*6
NUM_CSTEPS=NUM_R+NUM_WP
base='C:/Users/Usuario/Documents/Masterarbeit/code/'
dir=base+'gym-pybullet-drones-1.0.0/files/csvs/Experiments_v3/ATT/SO_100_const_eps0.3_ls.15.15/SO_100_const_rounds_100_AC70_35_70-04.12.2023_11.51'


data_x=pd.read_csv(dir+'/x0.csv')
data_y=pd.read_csv(dir+'/y0.csv')

chges=np.array((0, int((data_x.shape[0]+NUM_R)/NUM_CSTEPS)))
heights=np.array((1., 1.,1.))
start=0
rounds=float((data_x.shape[0]/(NUM_WP)+0.5))
#### BO #########################################
# # XY-trajectory
# plt.xlim([-0.34,0.34])
# plt.ylim([-0.18,0.18])

# XY-trajectory
for g in chges:
    # ground truth  
    # k=np.arange(0,NUM_WP,1)
    # x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
    # y_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2
    # plt.plot(x_true,y_true,label='true_value',linestyle='dashed')

    for i in range(g-start):
        if(i==(g-start-1)):
            k=np.arange(0,NUM_WP,1)
            x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
            y_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2
            plt.plot(x_true,y_true,label='true_value',linestyle='dashed')
            plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    data_y.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    label=('round'+str(i) + '(best)'), color='purple', linewidth=2)
        else:
            plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    data_y.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    label=('round'+str(i)), 
                    color=([max(1-2*i/(g-start),0),min(1, i/(g-start)),0,1-0.5*i/(g-start)]))

    start=g+2

    plt.xlim([-0.34,0.34])
    plt.ylim([-0.18,0.18])
    #plt.legend()
    plt.savefig(dir+'/trajectories_until'+str(g)+'.png')
    plt.show()

# Z-error
data_z=pd.read_csv(dir+'/z0.csv')
true_z=heights[0]*np.ones(data_z.iloc[:,1].shape)
for j in range(chges.size-1):
    true_z[chges[j]*NUM_CSTEPS:chges[j+1]*NUM_CSTEPS]=heights[j+1]*np.ones(true_z[chges[j]*NUM_CSTEPS:chges[j+1]*NUM_CSTEPS].shape)
error=np.abs(true_z-data_z.iloc[:,1])
plt.plot(data_z.iloc[:,0], error)
plt.savefig(dir+'/z_trajectory_error.png')
plt.show()

#### average cost per parameter combination ######
data_performance=pd.read_csv(dir+'/candidate_performance.csv').dropna(axis=0)
(-data_performance['cost']).plot()
plt.savefig(dir+'/cost.png')
plt.show()

#print(data_performance)



