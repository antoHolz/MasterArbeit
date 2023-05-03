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
import sys
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
from sklearn.neighbors import NearestNeighbors
import GPy
import utils
# import logging

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from SafeOpt import safeopt
sys.path.insert(0, ROOT_DIR+'/gym-pybullet-drones-1.0.0')
#sys.path.insert(0, "C:/Users/Usuario/Documents/Masterarbeit/code/gym-pybullet-drones-1.0.0")

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

import warnings

def hover(sec, j):
    for g in range(0,sec*env.SIM_FREQ, AGGR_PHY_STEPS):
        #### Step the simulation ###################################
        obs, reward, done, info = env.step(action)
        #### Compute control at the desired frequency ##############
        if g%CTRL_EVERY_N_STEPS == 0:
            action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                state=obs[str(j)]["state"],
                                                                target_pos=INIT_XYZS[j, 0:3],
                                                                # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                target_rpy=INIT_RPYS[j, :], 
                                                                #target_vel=TARGET_VEL[wp_counters[j], 0:3]
                                                                )
        ### Log the simulation ####################################
        logger.log(drone=j,
                timestamp=i/env.SIM_FREQ,
                state= obs[str(j)]["state"],
                control=np.hstack([TARGET_POS[j, 0:3], INIT_RPYS[j, :], np.zeros(6)])
                #control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                )
                #### Sync the simulation ###################################
        if ARGS.gui:
            sync(i, START, env.TIMESTEP)

if __name__ == "__main__":

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary or VisionAviary and DSLPIDControl')
    parser.add_argument('--drone',              default="cf2x",     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default="pyb_gnd_drag_dw",      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=False,       type=str2bool,     help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=120,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=60,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=40*12,      type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--baseline',           default=False,      type=str2bool,      help='Whether to do a baseline (no learning) or use the learning algorithm', metavar='')
    parser.add_argument('--max_learn_rounds',   default=15,         type=int,           help='Maximum learning rounds for SafeOpt algorithm', metavar='')
    parser.add_argument('--ET',                 default=True,       type=str2bool,      help='Whether to use Event-trigger', metavar='')
    parser.add_argument('--beta_type',          default="const",    type=str,           help='BEta type (boguovic, const)', metavar='')
    parser.add_argument('--experiment',         default="ATT",      type=str,           help='Experiment kind(ATT,PWM,GE,none)', metavar='')
    parser.add_argument('--repetitions',        default=1,         type=int,           help='Number of repetitions of the experiment', metavar='')
   
    ARGS = parser.parse_args()
    #physics originally "pyb"
    #### Change logging level
    #logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore") 

    H = 1.
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([[0, 0, H+i*H_STEP] for i in range(ARGS.num_drones)])#np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin(((i/6)*2*np.pi+np.pi/2)*2)/2, H+i*H_STEP] for i in range(ARGS.num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/ARGS.num_drones] for i in range(ARGS.num_drones)])
    AGGR_PHY_STEPS = int(ARGS.simulation_freq_hz/ARGS.control_freq_hz) if ARGS.aggregate else 1

    #### Initialize a 8-trayectory #############################
    PERIOD_8 = 6
    RETURN_TIME= 6
    PERIOD= PERIOD_8 + RETURN_TIME
    NUM_WP = int(ARGS.control_freq_hz * PERIOD_8)
    NUM_R= int(ARGS.control_freq_hz * RETURN_TIME)
    ROUND_STEPS= ARGS.simulation_freq_hz*PERIOD
    HOVER_STEPS=ARGS.simulation_freq_hz*RETURN_TIME
    TARGET_POS = utils.get_trajectory_xy(NUM_WP,NUM_R, INIT_XYZS[0,:], R) #default


    # TARGET_VEL = np.zeros((NUM_WP+NUM_R, 3))
    # for i in range(NUM_WP+NUM_R-1):
    #     TARGET_VEL[i+1, :]=(TARGET_POS[i+1]-TARGET_POS[i])/(ARGS.control_freq_hz)

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
    for g in range(ARGS.repetitions):
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
        
        ############################################################
        #### start PID parameter ###################################
        PID_coeff=np.array([[0.4, 0.4, 1.25],[.05, .05, .05],[.2, .2, .5], #PID xyz
        [70000., 70000., 60000.],[.0, .0, 500.],[20000., 20000., 12000.]]) #PID rpy
        candidate=np.array([PID_coeff[0:2][0:2]]).flatten()
        print(candidate)
        for i in range(ARGS.num_drones):
            ctrl[i].setPIDCoefficients(*PID_coeff) 

        #### Safety constraints ####################################
        delta=0.01
        #safe_start=np.array([0.4, 1.25,.05, .05])
        safe_start=np.array([0.4, 1.25,.05, .05])#np.array([0.973, 2.535,.013, .068]) #
        #### Learning init and constraints #########################   
        learn=True
        learn_counter=0
        max_learn_rounds=ARGS.max_learn_rounds

        #### Save init #############################################
        data_performance=pd.DataFrame(index=['theta','cost','norm_cost'])
        learn_at=pd.DataFrame(index=['timestep'])
        learn_trigger=pd.DataFrame(index=['rho','E_t'])

        ############################################################
        for j in range(ARGS.num_drones):
            hover(24, j)

        for i in range(0, int((ARGS.duration_sec)*env.SIM_FREQ), AGGR_PHY_STEPS):

            #### Step the simulation ###################################
            obs, reward, done, info = env.step(action)

            #### Compute control at the desired frequency ##############
            if i%CTRL_EVERY_N_STEPS == 0:

                #### Compute control for the current way point #############
                for j in range(ARGS.num_drones):
                    # #### Reset integral error #########################
                    # if(i%ROUND_STEPS==0):
                    #     ctrl[j].reset()
                    #### EXPERIMENTS ##################################
                    if(ARGS.experiment == "traj_1"):
                        if(i/ROUND_STEPS==0):
                            TARGET_POS=utils.get_trajectory_xz(NUM_WP,NUM_R, INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==20):
                            TARGET_POS=utils.bern_lemniscate(NUM_WP, NUM_R,INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==30):
                            TARGET_POS=utils.eck_lemniscate(NUM_WP, NUM_R,INIT_XYZS[0,:], R, .5*R)
                    
                    if(ARGS.experiment == "traj_2"):
                        if(i/ROUND_STEPS==0):
                            TARGET_POS=utils.get_trajectory_xy(NUM_WP,NUM_R, INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==20):
                            TARGET_POS=utils.bern_lemniscate_xy(NUM_WP, NUM_R,INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==30):
                            TARGET_POS=utils.eck_lemniscate_xy(NUM_WP, NUM_R,INIT_XYZS[0,:], R, .5*R)

                    if(ARGS.experiment == "traj_3"):
                        if(i/ROUND_STEPS==0):
                            TARGET_POS=utils.get_trajectory_xy(NUM_WP,NUM_R, INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==20):
                            TARGET_POS=utils.get_trajectory_xz(NUM_WP, NUM_R,INIT_XYZS[0,:], R)

                    if(ARGS.experiment=="PWM_ns"):
                        ### Update PWM ############################
                        if(i/ROUND_STEPS==0):
                            TARGET_POS=utils.get_trajectory_xz(NUM_WP,NUM_R, INIT_XYZS[0,:], R)
                        if(i/ROUND_STEPS==20):
                            ctrl[j].setPWM2RPM_scale_hw([[0.2685,0.2685,.2685,.245]])
                            hover(24,j)
                        # elif(i/ROUND_STEPS==40):
                        #     ctrl[j].setPWM2RPM_scale_hw([[0.2685,0.2685,.2685,.235]])
                        #     hover(24,j)
                        # elif(i/ROUND_STEPS==60):
                        #     ctrl[j].setPWM2RPM_scale_hw([[0.2685,0.2685,.2685,.230]])
                        #     hover(24,j)
                    if(ARGS.experiment=="PWM"):
                        ### Update PWM ############################
                        if(i/ROUND_STEPS==0):
                            TARGET_POS=utils.get_trajectory_xz(NUM_WP,NUM_R, INIT_XYZS[0,:], R)
                        if(i/ROUND_STEPS==20):
                            ctrl[j].setPWM2RPM_scale_hw(0.25)
                            hover(24,j)
                        # elif(i/ROUND_STEPS==30):
                        #     ctrl[j].setPWM2RPM_scale_hw(0.245)
                        #     hover(24,j)
                        # elif(i/ROUND_STEPS==45):
                        #     ctrl[j].setPWM2RPM_scale_hw(0.24)
                        #     hover(24,j)
                    if(ARGS.experiment=="ATT"):
                        # #### detuning AC ############################
                        if(i/ROUND_STEPS==20): 
                            PID_coeff=np.array([[0.4, 0.4, 1.25],[.05, .05, .05],[.2, .2, .5], #PID xyz
                            [42000., 42000., 36000.],[.0, .0, 500.],[20000., 20000., 12000.]]) #PID rpy
                            ctrl[j].setPIDCoefficients(*PID_coeff) 
                            hover(24,j)
                        elif(i/ROUND_STEPS==60):
                            PID_coeff=np.array([[0.4, 0.4, 1.25],[.05, .05, .05],[.2, .2, .5], #PID xyz
                            [70000., 70000., 60000.],[.0, .0, 500.],[20000., 20000., 12000.]]) #PID rpy
                            ctrl[j].setPIDCoefficients(*PID_coeff)
                            hover(24,j)
                    if(ARGS.experiment=="GE"):
                        ### Update soll trajectory ############################
                        if(i/ROUND_STEPS==20):
                            TARGET_POS=utils.get_trajectory_xy(NUM_WP,NUM_R, [0,0,.05], R)
                            hover(24,j)
                        elif(i/ROUND_STEPS==40):
                            TARGET_POS=utils.get_trajectory_xy(NUM_WP,NUM_R, [0,0,1.], R)
                            hover(24,j)

                    if(ARGS.experiment=="xy-xz-flip"):
                        
                        if(i/ROUND_STEPS==100):
                            TARGET_POS, TARGET_VEL, TARGET_ACC, TARGET_QUAT, TARGET_ANG_VEL=utils.get_trajectory_xyf(NUM_WP,NUM_R, INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==100):
                            TARGET_POS, TARGET_VEL, TARGET_ACC, TARGET_QUAT, TARGET_ANG_VEL=utils.get_trajectory_xzf(NUM_WP, NUM_R,INIT_XYZS[0,:], R)
                        elif(i/ROUND_STEPS==0):
                            TARGET_POS, TARGET_VEL, TARGET_ACC, TARGET_QUAT, TARGET_ANG_VEL=utils.get_flip_trajectory(0.9, PERIOD_8+RETURN_TIME-0.9, ARGS.control_freq_hz, H)
                            flip=True
                    
                    action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP,
                                                                        state=obs[str(j)]["state"],
                                                                        target_pos=TARGET_POS[wp_counters[j], 0:3],
                                                                        # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                                        # target_rpy=TARGET_QUAT[wp_counters[j], 0:3], 
                                                                        # target_vel=TARGET_VEL[wp_counters[j], 0:3],
                                                                        # target_rpy_rates=TARGET_ANG_VEL[wp_counters[j], 0:3]
                                                                        )
                    #print(action[str(j)])
     
                #### Position data for BO ###################################
                if((i%ROUND_STEPS)>HOVER_STEPS): #only count round, not hover steps               
                    if(i%ROUND_STEPS==(HOVER_STEPS+CTRL_EVERY_N_STEPS)):
                        X_ist=np.expand_dims(obs[str(j)]["state"][0:3],0)
                        #U= np.expand_dims(np.abs(action[str(j)]),0)
                    else:
                        X_ist=np.concatenate((X_ist, np.expand_dims(obs[str(j)]["state"][0:3],0)))
                        #U=np.concatenate((U, np.expand_dims(np.abs(action[str(j)]),0)))

                #### Calculate performance ##################################
                
                if(((i+CTRL_EVERY_N_STEPS)%ROUND_STEPS) == 0):
                    round_nr=int(i/(PERIOD*env.SIM_FREQ))
                    #### Calculate performance metric (cost) ################
                    nbrs=NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X_ist)
                    distances, indices = nbrs.kneighbors(TARGET_POS[NUM_R:])
                    #U=np.squeeze(U)
                    cost=-np.sum(np.sqrt(np.sqrt(distances))) #-5e-22*np.sum(np.dot(U,U.T)**2)

                    #### Cost normalization #################################
                    if(round_nr==0 ): 
                        cost_norm=int(-cost)*1.02
                    cost=np.expand_dims(np.expand_dims(cost,0),0) 
                    normalized_cost=cost/cost_norm

                    #### Save old candidate #################################
                    data_performance=data_performance.append({'theta':candidate.flatten(),
                    'cost':cost.squeeze(), 'norm_cost':normalized_cost.squeeze()}, ignore_index=True)
                    
                    #### Output of round performance infos ##################
                    print( "Round " +str(round_nr)+ "/" +str(int(ARGS.duration_sec/PERIOD)-1)
                            + " cost :" + str(cost.item()) 
                            + " ("+str((normalized_cost).item())+")" )   
                if(i>1 and (i%ROUND_STEPS) == 0 and (obs[str(j)]["state"][2]<.05 or cost<-300)):
                    break;
                    
                if not ARGS.baseline:
                    if((i%ROUND_STEPS) == 0):
                        ############################################################
                        #### Learning trigger ###################################### 
                        # Check if underlying function has changed too much 
                        if(learn_counter>=2 and ARGS.ET):
                            mu_t, var_t= gp.predict_noiseless(n_candidate)
                            pi_t=min((np.pi**2)*(learn_counter**2)/6, (np.pi**2)*(max_learn_rounds**2)/6)
                            rho_t=2*np.log(2*pi_t/delta)
                            omega_t=np.sqrt(2*noise_var*np.log(2*pi_t/delta))
                            E_t=(np.sqrt(rho_t)*np.sqrt(var_t)*3+omega_t)
                            #print(4*np.abs(normalized_cost-mu_t),E_t)

                            # #### Sanity check beta##################################
                            # if(normalized_cost<(mu_t+np.sqrt(var_t)*beta(learn_counter)) and 
                            # normalized_cost>(mu_t-np.sqrt(var_t)*beta(learn_counter))):
                            #     print('beta sane')
                            # else:
                            #     print('beta fail:  '+str(normalized_cost) + 
                            #     '  max:' +str((mu_t+np.sqrt(var_t)*beta(learn_counter))) +
                            #      '   min: '+str((mu_t-np.sqrt(var_t)*beta(learn_counter)))) 

                            if(4*np.abs(normalized_cost-mu_t)>E_t):
                                learn=True
                                learn_counter=1
                                cost_norm=int(-cost)*1.02
                                normalized_cost=cost/cost_norm

                                print("new learning triggered at round: "+ str(round_nr+1))
                                print(4*np.abs(normalized_cost-mu_t),E_t)
                                
                                learn_at=learn_at.append({'timestep':round_nr+1}, ignore_index=True)
                                learn_trigger=learn_trigger.append({'rho':rho_t,
                                'E_t':E_t},ignore_index=True)
                            
                        ############################################################
                        
                        if(learn):
                            #### BO ################################
                            #### Fit new GP ###################                      
                            if(learn_counter==1):
                                #### Measurement noise
                                noise_var = (0.01*normalized_cost.squeeze())** 2 #
                                #### Bounds on the input variables and normalization of theta
                                bounds = [(0, 1e1), (0, 1e1),#P
                                        (0, 1e1), (0, 1e1)] #I
                                theta_norm=[b[1]-b[0] for b in bounds]
                                n_candidate=np.divide(candidate.squeeze().flatten(),theta_norm)
                                n_candidate=np.expand_dims(n_candidate, 0)
                                
                                #### Prior mean
                                prior_mean= -1
                                def constant(num):
                                    return prior_mean
                                mf = GPy.core.Mapping(4,1)
                                mf.f = constant
                                mf.update_gradients = lambda a,b: None

                                #### Beta
                                if(ARGS.beta_type=="bog"):
                                    beta=utils.beta_bog
                                elif(ARGS.beta_type=="const"):
                                    beta=utils.beta_const
                                
                                #### Define Kernel
                                lengthscale=[0.15/theta_norm[0], 0.75/theta_norm[1],  
                                            0.025/theta_norm[2], 0.05/theta_norm[3]]
                                EPS=0.3
                                prior_std=max((1/3)*np.abs(prior_mean),prior_mean*(1-(1+0.1*beta(1))*(1+EPS))/beta(1))
                                kernel = GPy.kern.src.stationary.Matern52(input_dim=len(bounds), 
                                        variance=prior_std**2, lengthscale=lengthscale, ARD=4)
                                # kernel = GPy.kern.RBF(input_dim=len(bounds), variance=prior_std**2, 
                                # lengthscale=lengthscale, ARD=4)

                                #### The statistical model of our objective function
                                gp = GPy.models.GPRegression(n_candidate, normalized_cost, 
                                                            kernel, noise_var=noise_var, mean_function=mf)

                                mu_0, var_0= gp.predict_noiseless(n_candidate)
                                sigma_0=np.sqrt(var_0.squeeze())
                                J_min=(mu_0-beta(1)*sigma_0)*(1+EPS)
                                
                                opt = safeopt.SafeOptSwarm(gp, J_min, bounds=[(0,1) for b in bounds], 
                                                            threshold=0.2, beta=beta)

                                print("BETA: "+str(beta(1)) + "    PRIOR MEAN: "+ str(prior_mean) 
                                        +"     J_MIN: "+ str(J_min.item()) + "  J_NORM: "+ str(cost_norm))
                            #### Add new point to the GP model  ###############
                            elif(learn_counter>1):
                                
                                opt.add_new_data_point(n_candidate,  normalized_cost) 
                            ########################################################

                        #### Obtain next query point ########################### 
                            if(learn_counter==0):
                                candidate= safe_start #np.array([pi for pi in PID_coeff[0:2]]).flatten() #
                                print("START CANDIDATE: "+str(candidate))
                                
                            elif((learn_counter)<max_learn_rounds ): 
                                if((ARGS.ET and ARGS.max_learn_rounds==100) and 
                                (learn_counter>=10 and (learn_counter+1)%5==0)):
                                    id_check=data_performance['theta'].isna().sum()+round_nr-5
                                    candidate=data_performance.at[id_check,'theta']
                                    n_candidate=np.divide(candidate.squeeze(),theta_norm)
                                    print('test round')
                                else:
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
                        ####################################################################
                        learn_counter+=1
                                
                    if((i%ROUND_STEPS) == 0):
                        #### Set new PID parameters ################################
                        cand=np.array([candidate[0], candidate[0], candidate[1],
                                        candidate[2],candidate[2],candidate[3]])
                        for l in range(2):
                            PID_coeff[l]=np.reshape(cand[3*l:3*l+3].squeeze(),PID_coeff[l].shape)
                        ctrl[j].setPIDCoefficients(*PID_coeff)  
                

                ### Go to the next way point and loop #####################
                for j in range(ARGS.num_drones): 
                    wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP+NUM_R-1) else 0
            ### Log the simulation ####################################
            for j in range(ARGS.num_drones):
                logger.log(drone=j,
                        timestamp=i/env.SIM_FREQ,
                        state= obs[str(j)]["state"],
                        control=np.hstack([TARGET_POS[wp_counters[j], 0:3], INIT_RPYS[j, :], np.zeros(6)])
                        #control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                        )

            # ### Printout ##############################################
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
        os.environ["HOME"]=ROOT_DIR #most people don't need this
        logger.save()
        
        ## custom save with added performance metric
        NR_ROUNDS=str(int(ARGS.duration_sec/PERIOD))
        if ARGS.baseline:
            ALGORITHM="baseline"
        elif(ARGS.ET):
            ALGORITHM="ET_"+str(ARGS.max_learn_rounds)+"_"+ARGS.beta_type
        else:
            ALGORITHM="SO_"+str(ARGS.max_learn_rounds)+"_"+ARGS.beta_type
        
        FILENAME="Experiments_v3/"+ARGS.experiment+"/"+ALGORITHM+"/"+ALGORITHM+"_rounds_"+NR_ROUNDS
        logger.save_as_csv_w_performance(FILENAME, data_performance, learn_at, others=learn_trigger)

        # #### Plot the simulation results ###########################
        # if ARGS.plot:
        #     logger.plot()


