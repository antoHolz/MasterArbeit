#!/usr/bin/env python

import csv
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from pycrazyswarm import *

import sys
sys.path.insert(0, "/home/antonia")
from SafeOpt import safeopt
import GPy

def get_postion(phi, z):
    x = 2*np.cos(phi)
    y = 2*np.sin(2*phi) / 2
    return np.array([x, y, z])


def deg2rad(angle):
    return angle * np.pi / 180.

#### Prior mean
PRIOR_MEAN= -1
def constant(num):
    return PRIOR_MEAN


if __name__ == "__main__":
    
    #### Init swarm #####################################
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    TIMESCALE = 1.0

    #### Init learning ###################################
    learn=True
    learn_counter=0
    max_learn_rounds=3

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

    #### Lemniscate trajectory ###########################
    for rounds in range(3):
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
        cost=np.expand_dims(np.expand_dims(cost,0),0)
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
        candidate=np.array([0.4,0.4,1.25,0.05,0.05,0.05])#[P_x,P_y,P_z,I_x,I_y,I_z]
    
        if(learn):
            #### BO ##############################################
            #### Fit new GP ######################################                      
            if(learn_counter==0):
                #### Measurement noise
                noise_var = (0.01*n_cost.squeeze())** 2 #
                #### Bounds on the input variables and normalization of theta
                bounds = [(0, 2e0), (0, 2e0), (0, 2e0),#P
                        (0, 1e0), (0, 1e0), (0, 1e0)] #I
                theta_norm=[b[1]-b[0] for b in bounds]
                n_candidate=np.divide(candidate.squeeze().flatten(),theta_norm)
                n_candidate=np.expand_dims(n_candidate, 0)
                
                #### Prior mean function setup ##########
                mf = GPy.core.Mapping(6,1)
                mf.f = constant
                mf.update_gradients = lambda a,b: None
                
                #### Define Kernel ######################
                lengthscale=[0.25/theta_norm[0], 0.25/theta_norm[1], 0.25/theta_norm[2],  
                            0.025/theta_norm[3], 0.025/theta_norm[4], 0.025/theta_norm[5]]
                EPS=0.2
                def beta(learn_counter):
                    return 0.8*np.log(4*learn_counter)   #bogunovic 0.8*np.log(2*learn_counter)
                prior_std=max((1/3)*np.abs(PRIOR_MEAN),PRIOR_MEAN*(1-(1+0.1*beta(1))*(1+EPS))/beta(1))
                kernel = GPy.kern.src.stationary.Matern52(input_dim=len(bounds), 
                        variance=prior_std**2, lengthscale=lengthscale, ARD=6)
                # kernel = GPy.kern.RBF(input_dim=len(bounds), variance=prior_std**2, 
                # lengthscale=lengthscale, ARD=4)

                #### The statistical model of our objective function
                gp = GPy.models.GPRegression(n_candidate, n_cost, 
                                            kernel, noise_var=noise_var, mean_function=mf)

                mu_0, var_0= gp.predict_noiseless(n_candidate)
                sigma_0=np.sqrt(var_0.squeeze())
                J_min=(mu_0-beta(1)*sigma_0)*(1+EPS)
                
                opt = safeopt.SafeOptSwarm(gp, J_min, bounds=[(0,1) for b in bounds], 
                                            threshold=0.2, beta=beta, swarm_size=60)

                print("BETA: "+str(beta(1)) + "    PRIOR MEAN: "+ str(PRIOR_MEAN) 
                        +"     J_MIN: "+ str(J_min.item()) + "  J_NORM: "+ str(norm))
            #### Add new point to the GP model  ###############
            elif(learn_counter>0):
                
                opt.add_new_data_point(n_candidate,  n_cost) 
            ########################################################

        #### Obtain next query point ########################### 
                
            if((learn_counter+1)<max_learn_rounds ): #and (cost<=-80)):              
                n_candidate = opt.optimize()#ucb=True) 
                n_candidate=np.expand_dims(n_candidate, 0)
                candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                
                print("NEW CANDIDATE: "+str(candidate) + "  BETA: "+str(beta(learn_counter)))

            elif((learn_counter+1)==max_learn_rounds ):
                #### after last learn round, take best parameters #########
                n_candidate, _ = opt.get_maximum_S()
                n_candidate=np.expand_dims(n_candidate, 0)
                candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                
                print("BEST CANDIDATE: "+str(candidate))
                
            else:
                #### reset learning ########################################
                learn=False
        #### Set new candidate param ###################################
        cf.setParam("posCtlPid/xKp", candidate[0])
        cf.setParam("posCtlPid/yKp", candidate[1])
        cf.setParam("posCtlPid/zKp", candidate[2])
        cf.setParam("posCtlPid/xKi", candidate[3])
        cf.setParam("posCtlPid/yKi", candidate[4])
        cf.setParam("posCtlPid/zKi", candidate[5])

        ####################################################################
        learn_counter+=1





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
