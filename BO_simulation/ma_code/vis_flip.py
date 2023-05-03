import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import utils


R=.3
NUM_WP=int(120*0.9)
NUM_R=int(120*6)
NUM_CSTEPS=NUM_R+NUM_WP
base=os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
dir=base+'/gym-pybullet-drones-1.0.0/files/csvs/Experiments_v3/flip_mass/ET_15_const/ET_15_const_rounds_40-05.02.2023_18.08'


data_x=pd.read_csv(dir+'/x0.csv')
data_z=pd.read_csv(dir+'/z0.csv')
chges=np.array((20,int((data_x.shape[0]+NUM_R)/NUM_CSTEPS-0.5)))
rounds=int(data_x.shape[0]/(NUM_WP+NUM_R))
Z_0=1.0
start=0
#### BO #########################################
# # XY-trajectory
# plt.xlim([-0.34,0.34])
# plt.ylim([-0.18,0.18])

for index, g in enumerate(chges):
    for i in range(g-start):
        if(i==(g-start-1)):

            k=np.arange(0,NUM_WP,1)
            target_pos, _,_,_,_=utils.flip(0.9,2,200,1.)
            x_true=target_pos[:,0]
            z_true=target_pos[:,2]
            plt.plot(x_true,z_true,label='true_value',linestyle='dashed')
            plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    data_z.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    label=('round'+str(start+i) + '(best)'), color='purple', linewidth=2)
        else:
            if(i==0):
                plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    data_z.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    label=('round'+str(i)), 
                    color='red', linestyle='dashed')
            else:
                plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                        data_z.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                        label=('round'+str(i)), 
                        color=([max(1-2*i/(g-start),0),min(1, i/(g-start)),0,0.3]))
    start=g



    # plt.xlim([-0.1,0.8])
    # plt.ylim([Z_0-.8/2,Z_0+.5])
    #plt.legend()
    plt.savefig(dir+'/flip_traj_until'+str(g)+'.png')
    plt.show()


#### average cost per parameter combination ######
data_performance=pd.read_csv(dir+'/candidate_performance.csv').dropna(axis=0)
(-data_performance['cost']).plot()
plt.savefig(dir+'/cost.png')
plt.show()

#print(data_performance)

# for i in range(40):
#     plt.plot(data_x.iloc[:,1][240+NUM_CSTEPS*(i):NUM_CSTEPS*(i)+348], 
#             data_z.iloc[:,1][240+NUM_CSTEPS*(i):NUM_CSTEPS*(i)+348], 
#             label=('round'+str(i) + '(best)'), color='purple', linewidth=2)
# plt.show()

