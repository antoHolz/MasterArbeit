import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

def bern_lemniscate(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = 0.75*R * np.sqrt(2)* np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2)/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)
            **2+1) + start_pos[0], start_pos[1], R * np.sqrt(2)* np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2)* np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2
            )/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)**2+1) + start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    return TARGET_POS

def eck_lemniscate(NUM_WP, NUM_R, start_pos, width, height):
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        if(i>5*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = width-width*(i-5*NUM_WP/6)/(NUM_WP/6)+ start_pos[0],start_pos[1],height -height*(i-5*NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        elif(i>4*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] =  width+ start_pos[0], start_pos[1], -height + 2*height*(i-4*NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        elif(i>2*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = -width + width*(i-2*NUM_WP/6)/(NUM_WP/6)+ start_pos[0], start_pos[1], height - height*(i-2*NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        elif(i>1*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = -width+ start_pos[0], start_pos[1],-height+ 2*height*(i-NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        else:
            TARGET_POS[i+NUM_R, :] = -width*i/(NUM_WP/6)+ start_pos[0],start_pos[1],-height*i/(NUM_WP/6)+ start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
   
    return TARGET_POS

R=.3
NUM_WP=48*6
NUM_R=48*6
NUM_CSTEPS=NUM_R+NUM_WP
base=os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
dir=base+'/gym-pybullet-drones-1.0.0/files/csvs/Experiments_v3/PWM_ns/ET_15_const/ET_15_const_rounds_40-04.19.2023_15.44'


data_x=pd.read_csv(dir+'/x0.csv')
data_z=pd.read_csv(dir+'/z0.csv')

chges=np.array((15,int((data_x.shape[0]+NUM_R)/NUM_CSTEPS)))
y_true=np.array((0, 0,0))
start=0
rounds=float((data_x.shape[0]/(NUM_WP)+0.5))
Z_0=1.0
#### BO #########################################
# # XY-trajectory
# plt.xlim([-0.34,0.34])
# plt.ylim([-0.18,0.18])

# XY-trajectory
for index, g in enumerate(chges):
    # ground truth  
    # k=np.arange(0,NUM_WP,1)
    # x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
    # y_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2
    # plt.plot(x_true,y_true,label='true_value',linestyle='dashed')
    for i in range(g-start):
        if(i==(g-start-1)):
            k=np.arange(0,NUM_WP,1)
            x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
            z_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+Z_0
            plt.plot(x_true,z_true,label='true_value',linestyle='dashed')
            # if(index==0):
            #     k=np.arange(0,NUM_WP,1)
            #     x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
            #     z_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+Z_0
            #     plt.plot(x_true,z_true,label='true_value',linestyle='dashed')
            # elif(index==2):
            #     EL=eck_lemniscate(NUM_WP, NUM_R, [0,0,Z_0], R, .5*R)
            #     plt.plot(EL[:,0], EL[:,2],label='true_value',linestyle='dashed')
            
            # elif(index==1):
            #     BL=bern_lemniscate(NUM_WP, NUM_R, [0,0,Z_0], R)
            #     plt.plot(BL[:,0], BL[:,2],label='true_value',linestyle='dashed')
            plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    data_z.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    label=('round'+str(i) + '(best)'), color='purple', linewidth=2)
        else:
            if(i==2):
                plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    data_z.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                    label=('round'+str(i)), 
                    color='red', linestyle='dashed')
            else:
                plt.plot(data_x.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                        data_z.iloc[:,1][NUM_R+NUM_CSTEPS*(start+i):NUM_CSTEPS*(start+i+1)], 
                        label=('round'+str(i)), 
                        color=([max(1-2*i/(g-start),0),min(1, i/(g-start)),0,0.4]))

    start=g+2

    plt.xlim([-R*1.1,R*1.1])
    plt.ylim([Z_0-R*1.1/2,Z_0+R*1.1/2])
    #plt.legend()
    plt.savefig(dir+'/trajectories_until'+str(g)+'.png')
    plt.show()

# Y-error
data_y=pd.read_csv(dir+'/y0.csv')
true_y=y_true[0]*np.ones(data_y.iloc[:,1].shape)
for j in range(chges.size-1):
    true_y[chges[j]*NUM_CSTEPS:chges[j+1]*NUM_CSTEPS]=y_true[j+1]*np.ones(true_y[chges[j]*NUM_CSTEPS:chges[j+1]*NUM_CSTEPS].shape)
error=np.abs(true_y-data_y.iloc[:,1])
plt.plot(data_y.iloc[:,0], error)
plt.savefig(dir+'/z_trajectory_error.png')
plt.show()

#### average cost per parameter combination ######
data_performance=pd.read_csv(dir+'/candidate_performance.csv').dropna(axis=0)
(-data_performance['cost']).plot()
plt.savefig(dir+'/cost.png')
plt.show()

#print(data_performance)



