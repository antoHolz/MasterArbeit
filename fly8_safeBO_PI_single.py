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
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

import pandas as pd
import time
from datetime import datetime

import GPy
import safeopt
import torch

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
    parser.add_argument('--gui',                default=True,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=48,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=18,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    ARGS = parser.parse_args()

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
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin(((i/6)*2*np.pi+np.pi/2)*2)/2, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a 8-trayectory #############################
    PERIOD = 6
    NUM_WP = ARGS.control_freq_hz * PERIOD
    ROUND_STEPS= ARGS.simulation_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + INIT_XYZS[0, 0], R * np.sin(
            2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2 + INIT_XYZS[0, 1],  INIT_XYZS[0, 2]
    wp_counters = np.array([int((i * NUM_WP / 6) % NUM_WP) for i in range(ARGS.num_drones)])
    
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
    PID_coeff=np.array([[.01, .01, 0.1],[.05, .05, .05],[.2, .2, .5],[70000., 70000., 60000.],[.0, .0, 500.],[20000., 20000., 12000.]])
    # Default param: [[.4, .4, 1.25],[.05, .05, .05],[.2, .2, .5],[70000., 70000., 60000.],[.0, .0, 500.],[20000., 20000., 12000.]])
    candidate=np.expand_dims(PID_coeff[0:2].flatten(), 0)
    for i in range(ARGS.num_drones):
        ctrl[i].setPIDCoefficients(*PID_coeff) 

    # Theta=np.empty(candidate.shape, dtype=np.float64)
    # Y=np.empty((1,1), dtype=np.float64)  

    # Matrix for LQR cost
    Q=np.eye(13)*np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    R_coeff=0
    R=R_coeff*np.eye(4)

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
                                                                       target_rpy=INIT_RPYS[j, :]
                                                                       )
            #### x, u data for BO ##################
            X_ist=np.hstack((np.array(obs[str(j)]["state"][0:7]),np.array(obs[str(j)]["state"][10:16])))
            X_soll=np.hstack((np.array(TARGET_POS[wp_counters[j], 0:3]), np.zeros(10)))
            
            if(i%ROUND_STEPS==0):
                x=np.abs(np.expand_dims((X_ist-X_soll),0))
                u=np.abs(np.expand_dims(np.array(action[str(j)]),0))
            else:
                x=np.concatenate((x,np.expand_dims((X_ist-X_soll),0)))
                u=np.concatenate((u,np.expand_dims(np.array(action[str(j)]),0)))
              

            #### BO ################################
 
            if(((i+CTRL_EVERY_N_STEPS)%ROUND_STEPS) == 0):
                #### Calculate performance metric (cost) #### 
                if(int(i/(PERIOD*env.SIM_FREQ-0.5))<1):
                    xmax_0=np.max(x,axis=0)
                    umax=env.MAX_RPM # torch.max(u)
                    umin=0 # torch.min(u)
                for k in range(x.shape[1]): x[:,k]=(x[:,k])/xmax_0[k]
                for k in range(u.shape[1]): u[:,k]=(u[:,k]-umin)/(umax-umin)
                performance=-np.diagonal(np.matmul(np.matmul(x,Q),x.T) +np.matmul(np.matmul(u,R),u.T))
                
                if(int(i/(PERIOD*env.SIM_FREQ-0.5))<1): 
                    ynorm=np.max(np.abs(performance)) #linalg.norm(performance) 
                    y_0=performance.mean()/ynorm
                normalized_cost=np.expand_dims(np.expand_dims(np.array(performance.mean()/ynorm),0),0) 

                #### Save old candidate  and normalize ####### (ignore, save for csv)
                if(int(i/(PERIOD*env.SIM_FREQ-0.5))<1):
                    Theta=np.expand_dims(np.array(PID_coeff[0:2]).flatten(),0)
                    Y=normalized_cost 
                else:
                    Theta=np.concatenate((Theta, np.expand_dims(np.array(PID_coeff[0:2]).flatten(),0)),axis=0)
                    Y=np.concatenate((Y, normalized_cost),0)

                print( "Round " +str(int(i/(PERIOD*env.SIM_FREQ)))+ " of " +str(int(ARGS.duration_sec/PERIOD)-1)
                        + " cost :" + str(performance.mean().item()) 
                        + " ("+str((normalized_cost).item())+")" )
                
                #### Fit new GP ###################
                if(int(i/(PERIOD*env.SIM_FREQ-0.5))<1):
                    # Measurement noise
                    noise_var = 0.05*y_0 #0.05 ** 2 #
                    # Bounds on the inputs variable
                    bounds = [(1e-1, 0.5e0), (1e-1, 0.5e0),(.5, 2), (.02,.8), (.02, .8), (.02, .8)]
                    # Define Kernel
                    l=(.05,.05,0.05,.005,.005,0.005)
                    kernel = GPy.kern.RBF(input_dim=len(bounds), variance=0.2*y_0, lengthscale=l,
                        ARD=3)
                    # The statistical model of our objective function
                    print(candidate)
                    gp = GPy.models.GPRegression(candidate, normalized_cost, kernel, noise_var=noise_var)

                    # The optimization routine
                    opt = safeopt.SafeOptSwarm(gp, -.75, bounds=bounds, threshold=0.2, beta=2)

                else:
                    # Add new point to the GP model
                    opt.add_new_data_point(candidate,  normalized_cost) 

                # opt.plot(100, plot_3d=False)
                # plt.show()
                
                #### model data ###################
                # print(
                #     "lengthscale: " +str(gp.covar_module.base_kernel.lengthscale)+ 
                #     "noise: " + str(gp.likelihood.noise) 
                #     )

                #### Obtain next query point ##################               
                if(int(i/(PERIOD*env.SIM_FREQ-0.5))>=(int(ARGS.duration_sec/PERIOD)-2)):
                    #for last 2 rounds, take best model
                    candidate, _ = opt.get_maximum()
                    print("BEST CANDIDATE: "+str(candidate))
                else:
                    candidate = opt.optimize()  
                    print("NEW CANDIDATE: "+str(candidate))#+ "   acquisition value: "+str(acq_value.item()))

                #### Set new PID parameters ################################
                PID_coeff[0:2]=np.reshape(np.array(candidate.squeeze()),PID_coeff[0:2].shape)
                ctrl[j].setPIDCoefficients(*PID_coeff)        
                

            #### Go to the next way point and loop #####################
            for j in range(ARGS.num_drones): 
                wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

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
    candidates=pd.DataFrame(np.vstack((theta_save,np.squeeze(Y))))
    AQUISITION_F="st"
    PID_START="0.01"
    R_MATRIX=str(R_coeff)
    NR_ROUNDS=str(int(ARGS.duration_sec/PERIOD))
    FILENAME="pid_safeBO"+"_R_"+R_MATRIX+"_alpha_"+AQUISITION_F+"_start_PI_"+PID_START+"_rounds_"+NR_ROUNDS
    logger.save_as_csv_w_performance(FILENAME, candidates)

    #### Plot the simulation results ###########################
    if ARGS.plot:
        logger.plot()