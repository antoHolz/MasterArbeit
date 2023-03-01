#!/usr/bin/env python

import csv
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from pycrazyswarm import *

import sys
sys.path.insert(0, "/home/antonia")
from SafeOpt import safeopt

def get_postion(phi, z):
    x = 2*np.cos(phi)
    y = 2*np.sin(2*phi) / 2
    return np.array([x, y, z])


def deg2rad(angle):
    return angle * np.pi / 180.


if __name__ == "__main__":
    
    #### Init swarm #####################################
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    TIMESCALE = 1.0

    #### Init learning ###################################
    learn=True
    learn_counter=0

    #### Init save files #################################
    csv_file_name = "/home/antonia/trajectories.csv"  
    csv_performance= "/home/antonia/performance.csv"

    with open(csv_file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["x", "y", "z"])

    with open(csv_performance, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["P_x", "P_y", "P_z", "I_x", "I_y", "I_z", "cost", "norm_cost"])

    #### Init soll trajectory ############################
    Z_start=1.
    X_soll=np.zeros((360,3))
    for i, t in enumerate(range(90, 360 + 90)):
        new_angle = deg2rad(t)
        X_soll[i] = get_postion(new_angle,Z_start)
 
    #### Flight start ####################################
    allcfs.takeoff(targetHeight=Z_start, duration=2.0)
    timeHelper.sleep(2.5)

    for rounds in range(2):
        for i in range(360):
           for cf in allcfs.crazyflies:
                cf.cmdPosition(X_soll[i], 0,)           
                if(i==0):
                    X_ist=np.expand_dims(cf.position(),0)
                else:
                    X_ist=np.concatenate((X_ist, np.expand_dims(cf.position(),0)))
           timeHelper.sleep(0.05)
        
        #### Calculate performance ###########################
        nbrs=NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_ist)
        distances, indices = nbrs.kneighbors(X_soll)
        cost=-np.sum(np.sqrt(np.sqrt(distances)))
        if rounds==0:
            norm=int(-cost)+1
        n_cost=cost/norm
  
        #### Get PID parameters ##############################	  
        P_x=cf.getParam("posCtlPid/xKp")
        P_y=cf.getParam("posCtlPid/yKp")
        P_z=cf.getParam("posCtlPid/zKp")
        I_x=cf.getParam("posCtlPid/xKi")
        I_y=cf.getParam("posCtlPid/yKi")
        I_z=cf.getParam("posCtlPid/zKi")
    
        #### Save Trajectory #################################
        with open(csv_file_name, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            for row in list(X_soll):
                 wr.writerow(row)

        #### Save Performance #################################
        performance=list((P_x,P_y,P_z, I_x,I_y,I_z, cost,n_cost))
        with open(csv_performance, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(performance)
        print("candidate: ("+ str(P_x)+str(P_y)+str(P_z)+str(I_x)+
        str(I_y)+str(I_z)+")    cost: "+str(cost))
        
        #### Wait before starting new round ###################
        timeHelper.sleep(0.5)

    

        

    allcfs.land(targetHeight=0.05, duration=2.0)