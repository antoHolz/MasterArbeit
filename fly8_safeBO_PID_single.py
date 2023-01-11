"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` or `VisionAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python fly.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
# from datetime import datetime
# import pdb
# import math
# import random
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pandas as pd

import GPy
import safeopt
import logging

import sys
sys.path.insert(0, "C:/Users/Usuario/Documents/Masterarbeit/code/gym-pybullet-drones-1.0.0")

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=16,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

    #### Change logging level
    # logging.basicConfig(level=logging.INFO)


    # #### Initialize the simulation for circular trajectory #############################
    # H = .1
    # H_STEP = .05
    # R = .3
    # INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])
    # INIT_RPYS = np.array([[0, 0,  0] for i in range(ARGS.num_drones)])
    # AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    # #### Initialize a circular trajectory ######################
    #PERIOD = 10
    #NUM_WP = ARGS.control_freq_hz*PERIOD
    #TARGET_POS = np.zeros((NUM_WP,3))
    #for i in range(NUM_WP):
    #    TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    #wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(ARGS.num_drones)])
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])#np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin(((i/6)*2*np.pi+np.pi/2)*2)/2, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a 8-trayectory #############################
    PERIOD_8 = 6
    RETURN_TIME= 2
    PERIOD= PERIOD_8 + RETURN_TIME
    NUM_WP = ARGS.control_freq_hz * PERIOD_8
    NUM_R= ARGS.control_freq_hz * RETURN_TIME
    ROUND_STEPS= ARGS.simulation_freq_hz*PERIOD
    TRAJ_STEPS=ARGS.simulation_freq_hz*PERIOD_8

    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + INIT_XYZS[0, 0], R * np.sin(
            2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2 + INIT_XYZS[0, 1],  INIT_XYZS[0, 2]
    for i in range(NUM_R):
        TARGET_POS[i+NUM_WP, :] = INIT_XYZS[0,:]
    
    TARGET_VEL = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP+NUM_R-1):
        TARGET_VEL[i+1, :]=(TARGET_POS[i+1]-TARGET_POS[i])/(ARGS.control_freq_hz)

    wp_counters = np.array([int((i * (NUM_WP+NUM_R) / 6) % (NUM_WP+NUM_R)) for i in range(ARGS.num_drones)])
    
    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(ARGS.num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    # NUM_WP = ARGS.control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(ARGS.num_drones)])

    #### Create the environment with or without video capture ##
    if ARGS.vision: 
        env = VisionAviary(drone_model=ARGS.drone,
                           num_drones=ARGS.num_drones,
                           initial_xyzs=INIT_XYZS,
                           initial_rpys=INIT_RPYS,
                           physics=ARGS.physics,
                           neighbourhood_radius=10,
                           freq=ARGS.simulation_freq_hz,
                           aggregate_phy_steps=AGGR_PHY_STEPS,
                           gui=ARGS.gui,
                           record=ARGS.record_video,
                           obstacles=ARGS.obstacles
                           )
    else: 
        env = CtrlAviary(drone_model=ARGS.drone,
                         num_drones=ARGS.num_drones,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=ARGS.physics,
                         neighbourhood_radius=10,
                         freq=ARGS.simulation_freq_hz,
                         aggregate_phy_steps=AGGR_PHY_STEPS,
                         gui=ARGS.gui,
                         record=ARGS.record_video,
                         obstacles=ARGS.obstacles,
                         user_debug_gui=ARGS.user_debug_gui
                         )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=int(ARGS.simulation_freq_hz/AGGR_PHY_STEPS),
                    num_drones=ARGS.num_drones
                    )

    #### Initialize the controllers ############################
    if ARGS.drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]
    elif ARGS.drone in [DroneModel.HB]:
        ctrl = [SimplePIDControl(drone_model=ARGS.drone) for i in range(ARGS.num_drones)]

    #### Run the simulation ####################################
    CTRL_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/ARGS.control_freq_hz))
    action = {str(i): np.array([0,0,0,0]) for i in range(ARGS.num_drones)}
    START = time.time()
    
    #### BO in the simulation ##################################
    #start PID parameter
    PID_coeff=np.array([[0.4, 0.4, 1.25],[.05, .05, .05],[.2, .2, .5], #PID xyz
    [70000., 70000., 60000.],[.0, .0, 500.],[20000., 20000., 12000.]]) #PID rpy
    # Default param: [[.4, .4, 1.25],[.05, .05, .05],[.2, .2, .5],
    # [70000., 70000., 60000.],[.0, .0, 500.],[20000., 20000., 12000.]])
    candidate=PID_coeff[0:3].flatten()
    #candidate=np.expand_dims(candidate, 0)
    for i in range(ARGS.num_drones):
        ctrl[i].setPIDCoefficients(*PID_coeff) 
    print("START: " + str(candidate))

    # Matrix for LQR cost
    Q=np.eye(13)*np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    R_coeff=0
    Rm=R_coeff*np.eye(4)

    ############################################################

    for i in range(0, int(ARGS.duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)

        #### Compute control at the desired frequency ##############
        if i%CTRL_EVERY_N_STEPS == 0:
            #### Compute control for the current way point #############
            for j in range(ARGS.num_drones):
                action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                       state=obs[str(j)]["state"],
                                                                       target_pos=TARGET_POS[wp_counters[j], 0:3],
                                                                       # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                       target_rpy=INIT_RPYS[j, :], 
                                                                       #target_vel=TARGET_VEL[wp_counters[j], 0:3]
                                                                       )
            #### x, u data for BO ##################
            if(((i+1)%ROUND_STEPS)<TRAJ_STEPS): #only count round, not return steps
                X_ist=np.hstack((obs[str(j)]["state"][0:7],np.array(obs[str(j)]["state"][10:16])))
                X_soll=np.hstack((TARGET_POS[wp_counters[j], 0:3], np.zeros(10)))
                
                if(i%ROUND_STEPS==0):
                    x=np.abs(np.expand_dims((X_ist-X_soll),0))
                    u=np.abs(np.expand_dims(action[str(j)],0))
                else:
                    x=np.concatenate((x,np.expand_dims((X_ist-X_soll),0)))
                    u=np.concatenate((u,np.expand_dims(action[str(j)],0)))
              
            #### calculate performance ############# 
            if(((i+CTRL_EVERY_N_STEPS)>=TRAJ_STEPS) & (((i+CTRL_EVERY_N_STEPS)%ROUND_STEPS) == TRAJ_STEPS)):
                #### Calculate performance metric (cost) #### 
                if(int((i+CTRL_EVERY_N_STEPS)/TRAJ_STEPS)==1):
                    xmax_0=np.max(x,axis=0) #xmin=0 #also possible: xmax_0[0:3]=R
                    umax=env.MAX_RPM #umin=0
                for k in range(x.shape[1]): x[:,k]=(x[:,k])/xmax_0[k]
                for k in range(u.shape[1]): u[:,k]=(u[:,k])/(umax)
                performance=-np.diagonal(np.matmul(np.matmul(x,Q),x.T) +np.matmul(np.matmul(u,Rm),u.T))
                #Cost normalization
                if(int((i+CTRL_EVERY_N_STEPS)/TRAJ_STEPS)==1): 
                    ynorm=int(np.abs(performance.mean())+1) #4.5 #13.5 #np.max(np.abs(performance)) 
                cost=np.expand_dims(np.expand_dims(performance.mean(),0),0) 
                normalized_cost=cost/ynorm

                #### Save old candidate####### (ignore, save for csv)
                if(int((i+CTRL_EVERY_N_STEPS)/TRAJ_STEPS)==1):
                    Theta=np.expand_dims(candidate.flatten(),0)
                    Y=normalized_cost 
                    Y_abs=cost
                else:
                    Theta=np.concatenate((Theta, np.expand_dims(candidate.flatten(),0)),axis=0)
                    Y=np.concatenate((Y, normalized_cost),0)
                    Y_abs=np.concatenate((Y_abs, cost),0)
                #Output of round performance infos
                print( "Round " +str(int(i/(PERIOD*env.SIM_FREQ)))+ "/" +str(int(ARGS.duration_sec/PERIOD)-1)
                        + " cost :" + str(performance.mean().item()) 
                        + " ("+str((normalized_cost).item())+")" )
                print("vel: "+ str(np.sqrt(np.sum(np.array(obs[str(j)]["state"][10:13])**2))) +
                    "   rpm: "+ str(np.mean(action[str(j)])))
                
            #### BO ################################
            if((i%ROUND_STEPS) == TRAJ_STEPS):
                #### Fit new GP ###################
                if(int(i/ROUND_STEPS)==0):
                    #### Measurement noise
                    noise_var = 0.01*normalized_cost.squeeze() #0.05 ** 2 #
                    #### Bounds on the input variables and normalization of theta
                    bounds = [(0, 2e0), (0, 2e0), (0, 2e0), #P
                            (0, 1e0), (0, 1e0), (0, 1e0), #I
                            (0, 1e0), (0, 1e0), (0, 1e0)] #D
                    theta_norm=[b[1]-b[0] for b in bounds]
                    n_candidate=np.divide(candidate.squeeze().flatten(),theta_norm)
                    n_candidate=np.expand_dims(n_candidate, 0)
                    
                    #### Prior mean
                    prior_mean= -1
                    def constant(num):
                        return prior_mean
                    mf = GPy.core.Mapping(9,1)
                    mf.f = constant
                    mf.update_gradients = lambda a,b: None
                    
                    #### Define Kernel
                    lengthscale=[0.25/theta_norm[0], 0.25/theta_norm[1], 0.25/theta_norm[2],  
                                0.025/theta_norm[3], 0.025/theta_norm[4], 0.025/theta_norm[5],
                                0.025/theta_norm[0], 0.025/theta_norm[1], 0.025/theta_norm[2]]
                    prior_std=(1/3)*prior_mean
                    kernel = GPy.kern.src.stationary.Matern52(input_dim=len(bounds), 
                            variance=prior_std**2, lengthscale=lengthscale, ARD=9)
                    # kernel = GPy.kern.RBF(input_dim=len(bounds), variance=prior_std**2, 
                    # lengthscale=lengthscale, ARD=4)

                    #### The statistical model of our objective function
                    gp = GPy.models.GPRegression(n_candidate, normalized_cost, 
                                                kernel, noise_var=noise_var, mean_function=mf)

                    mu_0, var_0= gp.predict_noiseless(n_candidate)
                    sigma_0=np.sqrt(var_0.squeeze())
                    EPS=0.1
                    
                    def beta(round_nr):
                        return 0.8*np.log(4*round_nr)   #bogunovic
                    J_min=(mu_0-beta(1)*sigma_0)*(1+EPS)
                    
                    print("BETA: "+str(beta(1)) + "    PRIOR MEAN: "+ str(prior_mean) 
                            +"     J_MIN: "+ str(J_min.item()) + "  J_NORM: "+ str(ynorm))
                    opt = safeopt.SafeOptSwarm(gp, J_min, bounds=bounds, threshold=0.2, beta=beta, 
                                                swarm_size=20)
                #### Add new point to the GP model  ###############
                else:
                                    
                    opt.add_new_data_point(n_candidate,  normalized_cost) 

                #### GP-Plot ######################
                # if(int(i/(PERIOD*env.SIM_FREQ-0.5))>350):
                #     opt.plot(100, plot_3d=False)
                #     plt.show()

                # #### get maximum variance ####
                # x_maxi, std_maxi = opt.get_new_query_point('maximizers')
                # x_exp, std_exp = opt.get_new_query_point('expanders')
                # max_var=np.maximum(std_maxi.max(), std_exp.max())
                # print(max_var)

                #### Obtain next query point ##################               
                if(int(i/ROUND_STEPS+2)<(int(ARGS.duration_sec/PERIOD))):
                    n_candidate = opt.optimize()#ucb=True) 
                    candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                    round_nr=int(i/(PERIOD*env.SIM_FREQ))+1 
                    print("NEW CANDIDATE: "+str(candidate) + "  BETA: "+str(beta(round_nr)))#+ "   acquisition value: "+str(acq_value.item()))
                elif(int(i/ROUND_STEPS+2)==(int(ARGS.duration_sec/PERIOD))):
                    #for last round, take best parameters
                    n_candidate, _ = opt.get_maximum()
                    candidate=np.multiply(n_candidate.squeeze(),theta_norm)
                    print("BEST CANDIDATE: "+str(candidate))
                    
            if((i>=ROUND_STEPS) & ((i%ROUND_STEPS) == 0)):
                #### Set new PID parameters ################################
                for i in range(3):
                     PID_coeff[i]=np.reshape(candidate[3*i:3*i+3].squeeze(),PID_coeff[i].shape)
                ctrl[j].setPIDCoefficients(*PID_coeff)        
                

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones): 
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP+NUM_R-1) else 0

        #### Log the simulation ####################################
        for j in range(ARGS.num_drones):
            logger.log(drone=j,
                       timestamp=i/env.SIM_FREQ,
                       state= obs[str(j)]["state"],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        # if i%env.SIM_FREQ == 0:
        #     env.render()
        #     #### Print matrices with the images captured by each drone #
        #     if ARGS.vision:
        #         for j in range(ARGS.num_drones):
        #             print(obs[str(j)]["rgb"].shape, np.average(obs[str(j)]["rgb"]),
        #                   obs[str(j)]["dep"].shape, np.average(obs[str(j)]["dep"]),
        #                   obs[str(j)]["seg"].shape, np.average(obs[str(j)]["seg"])
        #                   )

        #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    os.environ["HOME"]='C:/Users/Usuario' #most people don't need this
    logger.save()
    #logger.save_as_csv("pid_BO")  # Optional CSV save
    #candidates.to_csv(os.environ.get('HOME')+"/Desktop/save-flight-"+"pid_BO"+"-"+datetime.now().strftime("%m.%d.%Y_%H.%M")+'/candidate_performance.csv')
    
    ## custom save with added performance metric
    theta_save=[str(Theta[i]) for i in range(Theta.shape[0])]
    candidates=pd.DataFrame(np.vstack((theta_save,np.squeeze(Y), np.squeeze(Y_abs))))
    AQUISITION_F="st"
    PID_START="0.4_0.05"
    R_MATRIX=str(R_coeff)
    NR_ROUNDS=str(int(ARGS.duration_sec/PERIOD))
    FILENAME="pid_safeBO"+"_matern52"+"_R_"+R_MATRIX+"_alpha_"+AQUISITION_F+"_start_PID_"+PID_START+"_rounds_"+NR_ROUNDS
    logger.save_as_csv_w_performance(FILENAME, candidates)

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()

