#!/usr/bin/env python3

import csv
import os
import numpy as np
import pandas as pd
import warnings
from sklearn.neighbors import NearestNeighbors
from SafeOpt import safeopt
from GPy.core import Mapping
from GPy.kern.src.stationary import Matern52
from GPy.models import GPRegression
from pycrazyswarm import *
from datetime import datetime

#import sys
#sys.path.insert(0, "/home/franka_panda")

#######################################################
#### Definitions for different kinds of lemniscate ####
def deg2rad(angle):
    return angle * np.pi / 180.

def get_trajectory_gerono(z, R):
    X_soll=np.zeros((360,3))
    for i, t in enumerate(range(90, 360 + 90)):
        phi = deg2rad(t)
        X_soll[i] = [R*np.cos(phi), R*np.sin(2*phi) / 2, z ]
    return X_soll

def get_trajectory_bernoulli(z, R):
    X_soll = np.zeros((360, 3))
    for i, t in enumerate(range(90, 360 + 90)):
        phi = deg2rad(t)
        X_soll[i] = [0.75*R * np.sqrt(2)* np.cos(phi )/(np.sin(phi)**2+1) ,\
        R * np.sqrt(2)* np.cos(phi )* np.sin(phi )/(np.sin(phi)**2+1),\
        z]
    return X_soll

def get_trajectory_edge(z, R):
    width=R
    height=R/2
    NUM_WP=360
    X_soll=np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        if(i>5*NUM_WP/6):
            X_soll[i, :] = width-width*(i-5*NUM_WP/6)/(NUM_WP/6),height -height*(i-5*NUM_WP/6)/(NUM_WP/6), z
        elif(i>4*NUM_WP/6):
            X_soll[i, :] =  width, -height + 2*height*(i-4*NUM_WP/6)/(NUM_WP/6), z
        elif(i>2*NUM_WP/6):
            X_soll[i, :] = -width + width*(i-2*NUM_WP/6)/(NUM_WP/6), height - height*(i-2*NUM_WP/6)/(NUM_WP/6), z
        elif(i>1*NUM_WP/6):
            X_soll[i, :] = -width,-height+ 2*height*(i-NUM_WP/6)/(NUM_WP/6), z
        else:
            X_soll[i, :] = -width*i/(NUM_WP/6),-height*i/(NUM_WP/6),z
    return X_soll

def get_trajectory_gerono_xz(z, R):
    X_soll=np.zeros((360,3))
    for i, t in enumerate(range(90, 360 + 90)):
        phi = deg2rad(t)
        X_soll[i] = [R*np.cos(phi), 0, z + R*np.sin(2*phi) / 2 ]
    return X_soll

def get_trajectory_bernoulli_xz(z, R):
    X_soll = np.zeros((360, 3))
    for i, t in enumerate(range(90, 360 + 90)):
        phi = deg2rad(t)
        X_soll[i] = [0.75*R * np.sqrt(2)* np.cos(phi )/(np.sin(phi)**2+1), 0,\
        R * np.sqrt(2)* np.cos(phi )* np.sin(phi )/(np.sin(phi)**2+1) + z]
    return X_soll

def get_trajectory_edge_xz(z, R):
    NUM_WP=360
    width=R
    height=R/2
    X_soll=np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        if(i>5*NUM_WP/6):
            X_soll[i, :] = width-width*(i-5*NUM_WP/6)/(NUM_WP/6),0, z +height -height*(i-5*NUM_WP/6)/(NUM_WP/6)
        elif(i>4*NUM_WP/6):
            X_soll[i, :] =  width, 0, z -height + 2*height*(i-4*NUM_WP/6)/(NUM_WP/6)
        elif(i>2*NUM_WP/6):
            X_soll[i, :] = -width + width*(i-2*NUM_WP/6)/(NUM_WP/6), 0, height - height*(i-2*NUM_WP/6)/(NUM_WP/6)+ z
        elif(i>1*NUM_WP/6):
            X_soll[i, :] = -width, 0, -height+ 2*height*(i-NUM_WP/6)/(NUM_WP/6) + z
        else:
            X_soll[i, :] = -width*i/(NUM_WP/6), 0, -height*i/(NUM_WP/6) + z
    return X_soll

#### Prior mean #######################################
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
    P_xy=0.4#0.2
    P_z=1.25 #1.
    I_xy=0.05 #0.05
    I_z=0.05 #0.05
    candidate=np.array([P_xy,P_z,I_xy,I_z])
    for cf in allcfs.crazyflies:
        cf.setParam("ctrlMel/kp_xy", P_xy) 
        cf.setParam("ctrlMel/ki_xy", I_xy) 
        cf.setParam("ctrlMel/kp_z", P_z) 
        cf.setParam("ctrlMel/ki_z", I_z) 
        # #### Sanity check ################
        # P_xy=cf.getParam("ctrlMel/kp_xy")
        # P_z=cf.getParam("ctrlMel/kp_z")
        # I_xy=cf.getParam("ctrlMel/ki_xy")
        # I_z=cf.getParam("ctrlMel/ki_z")
        print("START: "+str(candidate))

    #### Init learning ###################################
    #General
    learn=True
    learn_counter=0
    max_learn_rounds=9
    total_rounds=10
    continue_l=False
    exp_type="xyG"
    chg_at=[0, 9]
    Z=[1.]

    #Safeopt
    ls_px=0.15
    ls_pz=0.75
    ls_ix=0.025
    ls_iz=0.05
    #ls_d=0.05
    beta_type="const_2" #"bog"

    #Event-trigger
    delta=0.01
    trigger=True
    
    #### Init save files #################################
    if learn and (not trigger):
        dir="/home/franka_panda/Holz_drones/SOmel_{}_beta_{}_{}chges_{}_{}_".format(exp_type, beta_type, len(chg_at)-1, max_learn_rounds, total_rounds)+datetime.now().strftime("%m.%d_%H.%M")
    elif(trigger and learn):
        dir="/home/franka_panda/Holz_drones/ETLmel_{}_beta_{}_{}chges_{}_{}_".format(exp_type, beta_type, len(chg_at)-1, max_learn_rounds, total_rounds)+datetime.now().strftime("%m.%d_%H.%M")
    else:
        dir="/home/franka_panda/Holz_drones/Baseline_{}_{}chges_{}_{}_".format(exp_type, len(chg_at)-1, max_learn_rounds, total_rounds)+datetime.now().strftime("%m.%d_%H.%M")
    csv_trajectory = dir+"/trajectories.csv"
    csv_performance= dir+"/performance.csv"
    csv_trigger=dir+"/learn_trigger.csv"

    if not(os.path.exists(dir)):
        os.mkdir(dir)

    if continue_l:
        df=pd.read_csv(csv_performance)
    else:
        with open(csv_trajectory, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["x", "y", "z"])

        with open(csv_performance, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["P_xy","P_z","I_xy","I_z", "cost", "norm_cost"])

        with open(csv_trigger, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(["learn_at", "ET", "error"])

    #### Init soll trajectory ############################
    Z_start=Z[0]
    R=1
    X_soll=np.zeros((360,3))
    for i, t in enumerate(range(90, 360 + 90)):
        new_angle = deg2rad(t)
        X_soll = get_trajectory_gerono(Z[0],  R)   
        # X_soll = get_trajectory_bernoulli_xz(Z[0], R)   
        # X_soll = get_trajectory_edge_xz(Z[0], R)   
    if exp_type=="xy-xz":
        latency=.020
    else:
        latency=.017
 
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
        
    #### Lemniscate trajectory with learning ###################
    for rounds in range(total_rounds):
        ############################################################
        #### Learning trigger ###################################### 
        # Check if underlying function has changed too much 
        if(trigger and learn_counter>=2):
            mu_t, var_t= gp.predict_noiseless(n_candidate)
            pi_t=min((np.pi**2)*(learn_counter**2)/6, (np.pi**2)*(max_learn_rounds**2)/6)
            rho_t=2*np.log(2*pi_t/delta)
            omega_t=np.sqrt(2*noise_var*np.log(2*pi_t/delta))
            E_t=np.sqrt(rho_t)*3*np.sqrt(var_t)+omega_t
            err=4*np.abs(n_cost-mu_t)

            print(err.item(), E_t.item())
            if(err>E_t):
                #### Restart learning and norm #########################
                learn=True
                learn_counter=1
                norm=int(-cost)+1
                n_cost=cost/norm
                
                #### Save Trigger info #################################
                trigger_info=list((rounds+1,E_t,err))
                with open(csv_trigger, 'a', newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow(trigger_info)
                
                print("(new) learning triggered at round: "+ str(rounds+1))


                  
        if(learn):
            #### BO ##############################################
            #### Fit new GP ######################################                      
            if(learn_counter==1):
                #### Measurement noise
                noise_var = (0.016*n_cost.squeeze())** 2 #
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
                lengthscale=[ls_px/theta_norm[0], ls_pz/theta_norm[1], ls_ix/theta_norm[2], ls_iz/theta_norm[3]]#,  
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
                
                #### Add points from old dataframe to GP ##############
                if(continue_l):
                    for index in range(len(df.index)):
                        cnt=df.iloc[index]
                        candidate=np.array([cnt["P_xy"],cnt["P_z"],cnt["I_xy"], cnt["I_z"]])
                        n_candidate=np.divide(candidate.squeeze().flatten(),theta_norm)
                        n_candidate=np.expand_dims(n_candidate, 0)
                        n_cost=cnt["norm_cost"]
                        opt.add_new_data_point(n_candidate,  n_cost)
            #### Add new point to the GP model  ###############
            elif(learn_counter>1 or (learn_counter>0 and continue_l)):
                
                opt.add_new_data_point(n_candidate,  n_cost) 
            ########################################################

        #### Obtain next query point ########################### 
            if(learn_counter>=1):    
                if((learn_counter)<max_learn_rounds ): #and (cost<=-80)):     
                    # if(trigger and max_learn_rounds==100 and (learn_counter>8 and learn_counter%4==0)):
                    #     n_candidate=n_check
                    #     candidate=check
                    #     print("CHECK CANDIDATE: "+str(candidate) + "  BETA: "+str(beta(learn_counter)))

                    # else:         
                    n_candidate = opt.optimize()#ucb=True) 
                    n_candidate=np.expand_dims(n_candidate, 0)
                    candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                    print("NEW CANDIDATE: "+str(candidate) + "  BETA: "+str(beta(learn_counter)))
                        # if(learn_counter+1)%4==0:
                        #     n_check=n_candidate #Do df with startcheck the start candidate, get iloc[-2]
                        #     check=candidate
                        #     print('checksave')
                
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

                for i in range(1):
                    for cf in allcfs.crazyflies:
                        cf.cmdPosition([0,0,Z_start], 0,)           
                    timeHelper.sleep(.1) 
                    
                #### Set new candidate param ###################################
                cf.setParam("ctrlMel/kp_xy", P_xy)
                cf.setParam("ctrlMel/kp_z", P_z)
                cf.setParam("ctrlMel/ki_xy", I_xy)
                cf.setParam("ctrlMel/ki_z", I_z)


        ####################################################################
        learn_counter+=1   
        
        #### Trajectory change #####################################
        if(exp_type=="xyGBE"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Gerono lemniscate")
                    X_soll= get_trajectory_gerono(Z[0], R)   
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Bernoulli lemniscate")
                    X_soll = get_trajectory_bernoulli(Z[0], R)   
                if rounds == (chg_at[k]) and k==2:    
                    print("trajectory change! New trajectory: Edge lemniscate")
                    X_soll = get_trajectory_edge(Z[0], R)   
        elif(exp_type=="xyGB"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Gerono lemniscate")
                    X_soll= get_trajectory_gerono(Z[0], R)   
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Bernoulli lemniscate")
                    X_soll = get_trajectory_bernoulli(Z[0], R)   
        elif(exp_type=="xyG"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Gerono lemniscate")
                    X_soll= get_trajectory_gerono(Z[0], R)   
        elif(exp_type=="xyEB"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Bernoulli lemniscate")
                    X_soll = get_trajectory_bernoulli(Z[0], R)   
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Edge lemniscate")
                    X_soll = get_trajectory_edge(Z[0], R)   
        elif(exp_type=="xyBE"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Bernoulli lemniscate")
                    X_soll = get_trajectory_bernoulli(Z[0], R)   
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Edge lemniscate")
                    X_soll = get_trajectory_edge(Z[0], R)   
        elif(exp_type=="xyGE"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Gerono lemniscate")
                    X_soll = get_trajectory_gerono(Z[0], R)   
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Edge lemniscate")
                    X_soll = get_trajectory_edge(Z[0], R)   
        elif(exp_type=="xzBE"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Bernoulli lemniscate")
                    X_soll = get_trajectory_bernoulli_xz(Z[0], R)   
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Edge lemniscate")
                    X_soll = get_trajectory_edge_xz(Z[0], R)   
        elif(exp_type=="xzGE"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Gerono lemniscate")
                    X_soll = get_trajectory_gerono_xz(Z[0], R)   
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Edge lemniscate")
                    X_soll = get_trajectory_edge_xz(Z[0], R)   
        elif(exp_type=="xy-xz"):
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]) and k==0:    
                    print("trajectory change! New trajectory: Gerono_xy lemniscate")
                    X_soll = get_trajectory_gerono(Z[0], R)   
                if rounds == (chg_at[k]) and k==1:    
                    print("trajectory change! New trajectory: Gerono_xz lemniscate")
                    X_soll = get_trajectory_gerono_xz(Z[0], R)   
        # elif(exp_type=="ATT"):
        elif(exp_type=="GroundEffect"):
            #### Ground effect #########################################
            for k in range(len(chg_at)):
                if rounds == (chg_at[k]-1):
                    for i in range(5):
                        for cf in allcfs.crazyflies:
                            cf.cmdPosition([0,0,Z_start], 0,)           
                        timeHelper.sleep(.1)      
                    print("height change!")
                    X_soll[:,2]=np.ones(360)*Z[k+1]
                    for i in range(20):
                        for cf in allcfs.crazyflies:
                            cf.cmdPosition([0,0,Z[k+1]+(Z_start-Z[k+1])*(19-i)/20], 0,)           
                        timeHelper.sleep(.1)
                    Z_start=Z[k+1]
                    for i in range(10):
                        for cf in allcfs.crazyflies:
                            cf.cmdPosition([0,0,Z_start], 0,)           
                        timeHelper.sleep(.1)   
            
        #### Do lemniscate ###################################
        for i in range(360):
           for cf in allcfs.crazyflies:
                cf.cmdPosition(X_soll[i], 0,)           
                if(i==0):
                    X_ist=np.expand_dims(cf.position(),0)
                else:
                    X_ist=np.concatenate((X_ist, np.expand_dims(cf.position(),0)))
           timeHelper.sleep(latency)
        
        #### Calculate performance ###########################
        nbrs=NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_ist)
        distances, indices = nbrs.kneighbors(X_soll)
        cost=-np.sum(np.sqrt(np.sqrt(distances)))
        cost=np.expand_dims(np.expand_dims(cost,0),0)
        if rounds==0 :
            norm=-cost*1.02
        n_cost=cost/norm
        print("Round: "+str(rounds+1)+"/"+str(total_rounds) +"    Cost: "+str(cost.item()))
        

        #### Return to start ##################################
        for i in range(25):
            for cf in allcfs.crazyflies:
                cf.cmdPosition([0,0,Z_start], 0,)           
            timeHelper.sleep(.1)                
        
        #### Save Trajectory #################################
        with open(csv_trajectory, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            for row in list(X_ist):
                 wr.writerow(row)

        #### Save Performance #################################
        performance=list((P_xy,P_z,I_xy,I_z, np.squeeze(cost).item(),np.squeeze(n_cost).item()))
        with open(csv_performance, 'a', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(performance)

        

    allcfs.land(targetHeight=0.05, duration=2.0)

