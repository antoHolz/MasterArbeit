#!/usr/bin/env python3

import csv
import os
import numpy as np
import warnings
from sklearn.neighbors import NearestNeighbors
from SafeOpt import safeopt
from GPy.core import Mapping
from GPy.kern.src.stationary import Matern52
from GPy.models import GPRegression
from pycrazyswarm import *

#import sys
#sys.path.insert(0, "/home/franka_panda")


def get_postion(phi, z):
    x = np.cos(phi)
    y = np.sin(2*phi) / 2
    return np.array([x, y, z])


def deg2rad(angle):
    return angle * np.pi / 180.

#### Prior mean
PRIOR_MEAN= -1
def constant(num):
    return PRIOR_MEAN


if __name__ == "__main__":
    
    #### Init swarm ######################################
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    TIMESCALE = 1.0
    
    #### Set start parameters ############################
    for cf in allcfs.crazyflies:
        cf.setParam("ctrlMel/kp_xy", 0.4) #0.4
        cf.setParam("ctrlMel/ki_xy", 0.05) #0.05
        cf.setParam("ctrlMel/kp_z", 1.25) #1.25
        cf.setParam("ctrlMel/ki_z", 0.05) #0.05
    #    cf.setParam("ctrlMel/kd_xy", 0.2)

    #### Init learning ###################################
    learn=True
    learn_counter=1
    max_learn_rounds=9
    total_rounds=max_learn_rounds+1
    ls_p=0.25
    ls_i=0.05
    beta_type="bog" #"bog" "const_2" "const_anynumber"

    #### Init save files #################################
    dir="/home/franka_panda/Holz_drones/learn_sqrt__cardboard_h=.1_b4"#Mel_ls{}_{}_beta_{}_{}_noise0.1_sqrt".format(ls_p, ls_i,beta_type,total_rounds)  
    csv_file_name = dir+"/trajectories.csv"
    csv_performance= dir+"/performance.csv"

    if not(os.path.exists(dir)):
        os.mkdir(dir)

    with open(csv_file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["x", "y", "z"])

    with open(csv_performance, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["P_xy","P_z","I_xy","I_z", "cost", "norm_cost"])

    #### Init soll trajectory ############################
    Z_start=.1
    X_soll=np.zeros((360,3))
    for i, t in enumerate(range(90, 360 + 90)):
        new_angle = deg2rad(t)
        X_soll[i] = get_postion(new_angle,Z_start)
 
    #### Flight start ####################################
    allcfs.takeoff(targetHeight=Z_start, duration=2.0)
    timeHelper.sleep(2.5)
    for i in range(360):
           for cf in allcfs.crazyflies:
                cf.cmdPosition(X_soll[i], 0,)           
           timeHelper.sleep(0.017)

    for i in range(25):
        for cf in allcfs.crazyflies:
            cf.cmdPosition([0,0,Z_start], 0,)           
        timeHelper.sleep(.1)
    #### Lemniscate trajectory ###########################
    for rounds in range(total_rounds):
        for i in range(360):
           for cf in allcfs.crazyflies:
                cf.cmdPosition(X_soll[i], 0,)           
                if(i==0):
                    X_ist=np.expand_dims(cf.position(),0)
                else:
                    X_ist=np.concatenate((X_ist, np.expand_dims(cf.position(),0)))
           timeHelper.sleep(0.017)
        
        #### Get PID parameters ##############################
        if rounds == 0:	  
            P_xy=cf.getParam("ctrlMel/kp_xy")
            P_z=cf.getParam("ctrlMel/kp_z")
            I_xy=cf.getParam("ctrlMel/ki_xy")
            I_z=cf.getParam("ctrlMel/ki_z")
            candidate=np.array([P_xy,P_z,I_xy,I_z])# [0.4,0.4,1.25,0.05,0.05,0.05]
            print("START: "+str(candidate))
        
        #### Calculate performance ###########################
        nbrs=NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_ist)
        distances, indices = nbrs.kneighbors(X_soll)
        cost=-np.sum(np.sqrt(distances))
        cost=np.expand_dims(np.expand_dims(cost,0),0)
        if rounds==0:
            norm=int(-cost)+1
        n_cost=cost/norm
        print("Round: "+str(rounds+1)+"/"+str(total_rounds) +"    Cost: "+str(cost))
        
        #### Save Trajectory #################################
        with open(csv_file_name, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            for row in list(X_ist):
                 wr.writerow(row)

        #### Save Performance #################################
        performance=list((P_xy,P_z,I_xy,I_z, np.squeeze(cost).item(),np.squeeze(n_cost).item()))
        with open(csv_performance, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(performance)

        #### Return to start ##################################
        for i in range(25):
            for cf in allcfs.crazyflies:
                cf.cmdPosition([0,0,Z_start], 0,)           
            timeHelper.sleep(.1)
          
        if(learn):
            #### BO ##############################################
            #### Fit new GP ######################################                      
            if(learn_counter==1):
                #### Measurement noise
                noise_var = (0.01*n_cost.squeeze())** 2 #
                #### Bounds on the input variables and normalization of theta
                bounds = [(0, 1e1), (0, 1e1), (0, 1e1),  (0, 1e1)]#
                theta_norm=[b[1]-b[0] for b in bounds]
                n_candidate=np.divide(candidate.squeeze().flatten(),theta_norm)
                n_candidate=np.expand_dims(n_candidate, 0)
                
                #### Prior mean function setup #######################
                mf = Mapping(4,1)
                mf.f = constant
                mf.update_gradients = lambda a,b: None
                
                #### Define Kernel ###################################
                lengthscale=[ls_p/theta_norm[0], ls_p/theta_norm[1], ls_i/theta_norm[2], ls_i/theta_norm[3]]#,  
                #            0.05/theta_norm[3], 0.05/theta_norm[4], 0.05/theta_norm[5]]
                EPS=0.2
                if(beta_type=="bog"):
                    def beta(learn_counter):
                        return 0.8*np.log(4*learn_counter)   #bogunovic 0.8*np.log(2*learn_counter)
                elif(beta_type.split("_")[0]=="const"):
                    def beta(learn_counter):
                        return float(beta_type.split("_")[1])
                prior_std=max((1/3)*np.abs(PRIOR_MEAN),PRIOR_MEAN*(1-(1+0.1*beta(1))*(1+EPS))/beta(1))
                kernel = Matern52(input_dim=len(bounds), 
                        variance=prior_std**2, lengthscale=lengthscale, ARD=4)
                # kernel = GPy.kern.RBF(input_dim=len(bounds), variance=prior_std**2, 
                # lengthscale=lengthscale, ARD=4)

                #### The statistical model of our objective function
                gp = GPRegression(n_candidate, n_cost, 
                                            kernel, noise_var=noise_var, mean_function=mf)

                mu_0, var_0= gp.predict_noiseless(n_candidate)
                sigma_0=np.sqrt(var_0.squeeze())
                J_min=(mu_0-beta(1)*sigma_0)*(1+EPS)
                
                opt = safeopt.SafeOptSwarm(gp, J_min, bounds=[(0,1) for b in bounds], 
                                            threshold=0.2, beta=beta)# swarm_size=60

                print("BETA: "+str(beta(1)) + "    PRIOR MEAN: "+ str(PRIOR_MEAN) 
                        +"     J_MIN: "+ str(J_min.item()) + "  J_NORM: "+ str(norm))
            #### Add new point to the GP model  ###############
            elif(learn_counter>1):
                
                opt.add_new_data_point(n_candidate,  n_cost) 
            ########################################################

        #### Obtain next query point ########################### 
                
            if((learn_counter)<max_learn_rounds ): #and (cost<=-80)):              
                n_candidate = opt.optimize()#ucb=True) 
                n_candidate=np.expand_dims(n_candidate, 0)
                candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                
                print("NEW CANDIDATE: "+str(candidate) + "  BETA: "+str(beta(learn_counter)))

            elif((learn_counter)==max_learn_rounds ):
                #### after last learn round, take best parameters #########
                n_candidate, _ = opt.get_maximum_S()
                n_candidate=np.expand_dims(n_candidate, 0)
                candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                
                print("BEST CANDIDATE: "+str(candidate))
                
            else:
                #### reset learning ########################################
                learn=False
            P_xy=float(candidate[0])
            P_z=float(candidate[1])
            I_xy=float(candidate[2])
            I_z=float(candidate[3])

                
            #### Set new candidate param ###################################
            cf.setParam("ctrlMel/kp_xy", P_xy)
            cf.setParam("ctrlMel/kp_z", P_z)
            cf.setParam("ctrlMel/ki_xy", I_xy)
            cf.setParam("ctrlMel/ki_z", I_z)


        ####################################################################
        learn_counter+=1   
        

    allcfs.land(targetHeight=0.05, duration=2.0)

