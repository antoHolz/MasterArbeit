import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import utils

R=.3
NUM_WP=48*6
NUM_R=48*6
NUM_CSTEPS=NUM_R+NUM_WP

Z_0=0
#### BO #########################################
# # XY-trajectory

# def eck_lemniscate(NUM_WP, NUM_R, start_pos, width, height):
#     TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
#     for i in range(NUM_WP):
#         if(i>5*NUM_WP/6):
#             TARGET_POS[i+NUM_R, :] = width-width*(i-5*NUM_WP/6)/(NUM_WP/6),start_pos[1],height -height*(i-5*NUM_WP/6)/(NUM_WP/6)
#         elif(i>4*NUM_WP/6):
#             TARGET_POS[i+NUM_R, :] =  width, start_pos[1], -height + 2*height*(i-4*NUM_WP/6)/(NUM_WP/6)
#         elif(i>2*NUM_WP/6):
#             TARGET_POS[i+NUM_R, :] = -width + width*(i-2*NUM_WP/6)/(NUM_WP/6), start_pos[1], height - height*(i-2*NUM_WP/6)/(NUM_WP/6)
#         elif(i>1*NUM_WP/6):
#             TARGET_POS[i+NUM_R, :] = -width, start_pos[1],-height+ 2*height*(i-NUM_WP/6)/(NUM_WP/6)
#         else:
#             TARGET_POS[i+NUM_R, :] = -width*i/(NUM_WP/6),start_pos[1],-height*i/(NUM_WP/6)
    
#     for i in range(NUM_R):
#         TARGET_POS[i, :] = start_pos
#     return TARGET_POS

# def bern_lemniscate(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
#     #Lemniskate von Gerono
#     TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
#     for i in range(NUM_WP):
#         TARGET_POS[i+NUM_R, :] = 0.75*R * np.sqrt(2)* np.cos((i / NUM_WP) * (
#             2 * np.pi) + np.pi / 2)/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)
#             **2+1) + start_pos[0], start_pos[1], R * np.sqrt(2)* np.cos((i / NUM_WP) * (
#             2 * np.pi) + np.pi / 2)* np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2
#             )/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)**2+1) +start_pos[2]
#     for i in range(NUM_R):
#         TARGET_POS[i, :] = start_pos
#     return TARGET_POS
    
# EL=utils.eck_lemniscate(NUM_WP, NUM_R, [0,0,Z_0], R, .5*R)
# plt.plot(EL[:,0], EL[:,2])

# BL=utils.bern_lemniscate(NUM_WP, NUM_R, [0,0,Z_0], R)
# plt.plot(BL[:,0], BL[:,2])

# k=np.arange(0,NUM_WP,1)
# x_true=R * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
# z_true=R * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+Z_0
# plt.plot(x_true,z_true,label='true_value',linestyle='dashed')

# LM=utils.get_trajectory_xy(NUM_WP, NUM_R, [0,0,Z_0], R)
# LM[abs(LM)<1e-5]=0.0
# df=pd.DataFrame(LM)
# df.to_csv('traj.csv')

# x_true=.3 * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
# z_true=.4 * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+Z_0
# plt.plot(x_true,z_true,label='true_value',linestyle='dashed')

# x_true=.3 * np.cos((k / NUM_WP) * (2 * np.pi) + np.pi / 2)
# z_true=.6 * np.sin(2*((k / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+Z_0
# plt.plot(x_true,z_true,label='true_value',linestyle='dashed')

# plt.xlim([-.3,.3])
# plt.ylim([-.15,+.15])
# plt.show()

# k=3.16e-10*(60/2/np.pi)**2  
# print(np.sqrt(1/k)*30/np.pi)
utils.get_flip_trajectory(0.9, 72)
