#!/usr/bin/env python

import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pycrazyswarm import *

import sys
sys.path.insert(0, "/home")
from SafeOpt import safeopt

Z = 1.


def get_postion(phi):
    x = 2*np.cos(phi)
    y = 2*np.sin(2*phi) / 2
    return np.array([x, y, Z])


def deg2rad(angle):
    return angle * np.pi / 180.


if __name__ == "__main__":
    csv_file_name = "trajectories.csv"  # tajectories_wind
    csv_performance= "performance.csv"

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    TIMESCALE = 1.0

    allcfs.takeoff(targetHeight=Z, duration=2.0)
    timeHelper.sleep(2.5)

    # print parameters
    # for cf in allcfs.crazyflies:
    #     print(cf.getParam("posCtlPid/xKp"))
    #     print(cf.getParam("posCtlPid/xKi"))
    #     print(cf.getParam("posCtlPid/xKd"))
    #     print(cf.getParam("posCtlPid/yKp"))
    #     print(cf.getParam("posCtlPid/yKi"))
    #     print(cf.getParam("posCtlPid/yKd"))

    with open(csv_file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["x", "y", "z"])

    with open(csv_performance, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["P_x", "P_y", "P_z", "I_x", "I_y", "I_z", "cost", "norm_cost"])

    #init soll trajectory
    X_soll=np.zeros(360)
    for i, t in enumerate(range(90, 360 + 90)):
        new_angle = deg2rad(t)
        X_soll[i] = get_postion(new_angle)

    for i in range(360):
        for cf in allcfs.crazyflies:
            cf.cmdPosition(X_soll[i], 0,)
            position=cf.position()
            with open(csv_file_name, 'a', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(position)
            
            if(i==0):
                X_ist=np.expand_dims(position,0)
            else:
                X_ist=np.concatenate((X_ist, np.expand_dims(position,0)))
        
        timeHelper.sleep(0.05)

    #calculate performance
    nbrs=NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_ist)
    distances, indices = nbrs.kneighbors(X_soll)
    cost=-np.sum(np.sqrt(np.sqrt(distances)))

    

        

    allcfs.land(targetHeight=0.05, duration=2.0)