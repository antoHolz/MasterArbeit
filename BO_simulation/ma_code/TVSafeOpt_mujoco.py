import mujoco
import glfw
from scipy.spatial.transform import Rotation
import numpy as np
import sys
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import GPy
import utils
from utils import str2bool

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)
from SafeOpt import safeopt
sys.path.insert(0, ROOT_DIR+'/crazyflie_backflipping')
from ctrl.GeomControl import GeomControl
import time
from assets.util import sync, quat_conj, quat_mult
from assets.logger import Logger
import scipy.interpolate as si



def main():
    # Reading model data
    model = mujoco.MjModel.from_xml_path(ROOT_DIR+"/crazyflie_backflipping/assets/cf2.xml")
    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1280, 720, "Crazyflie in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 170, -30
    cam.lookat, cam.distance = [0, 0, 1], 2

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=30)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)
    ### Controller
    controller = GeomControl(model, data)

    ### Trajectory
    
    pos_ref_points = np.loadtxt(ROOT_DIR+"/crazyflie_backflipping/assets/pos_ref.csv", delimiter=',')
    pos_ref_time = np.linspace(0, 0.9, pos_ref_points.shape[0])
    flip_traj = [si.splrep(pos_ref_time, pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, 0*pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, pos_ref_points[:, 1], k=3)]
    quat_points = -1.99999 / (1 + np.exp(-20*(pos_ref_time-0.9/2))) + 1.99999/2
    rot_traj = si.splrep(pos_ref_time, quat_points, k=3)

    target_quat = np.array([1, 0, 0, 0])
    target_ang_vel = np.zeros(3)
    psi = 0

    timestep = 0.005
    simulation_step = 0.005
    control_step = 0.005
    graphics_step = 0.02
    episode_length = 12.9

    POS=utils.get_trajectory_xy(int(6/control_step),int(2/control_step), [0,0,1], .3)
    logger = Logger(episode_length, control_step)
    start = time.time()

    for i in range(int(episode_length / control_step)):
        simtime = data.time

        if simtime < 8:
            target_pos = POS[i] # np.array([0, 0, 1]) # 
            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)
        elif simtime < 8.9:
            eval_time = (simtime - 8.0)
            target_pos = np.array([si.splev(eval_time, flip_traj[i]) for i in range(3)])
            target_pos[2] = target_pos[2] + 1
            target_vel = np.array([si.splev(eval_time, flip_traj[i], der=1) for i in range(3)])
            target_acc = np.array([si.splev(eval_time, flip_traj[i], der=2) for i in range(3)])
            q0 = si.splev(eval_time, rot_traj)
            q2 = np.sqrt(1 - q0**2)
            target_quat = np.array([q0, 0, q2, 0])
            dq0 = si.splev(eval_time, rot_traj, der=1)
            dq2 = - dq0 * q0 / q2
            target_quat_vel = np.array([dq0, 0, dq2, 0])
            target_ang_vel = np.roll((2 * quat_mult(quat_conj(np.roll(target_quat, -1)), np.roll(target_quat_vel, -1)))[0:3], -1)

            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl, eta_R, mu_R, psi, rot_e = controller.compute_att_control(pos, quat, vel, ang_vel, target_pos, target_vel, target_acc,
                                                       target_quat=target_quat, target_ang_vel=target_ang_vel)
        else:
            target_pos = np.array([0, 0, 1])
            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)

        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
        state = np.hstack([target_pos, pos, Rotation.from_quat((np.roll(quat, -1))).as_euler('xyz'),
                           Rotation.from_quat((np.roll(target_quat, -1))).as_euler('xyz'), ang_vel, target_ang_vel,
                           quat, psi])
        logger.log(timestamp=simtime, state=state, control=data.ctrl)

        if i % (graphics_step / control_step) == 0:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
            mujoco.mjr_render(viewport, scn, con)

            glfw.swap_buffers(window)
            glfw.poll_events()

            # sync with wall-clock time
            sync(i, start, control_step)

            if glfw.window_should_close(window):
                break

    glfw.terminate()
    logger.plot()
    logger.save_as_csv(comment='hi')


if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and GeomControl')
    parser.add_argument('--drone',              default="cf2x",     type=str,    help='Drone model (default: CF2X)', metavar='')
    parser.add_argument('--num_drones',         default=1,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--vision',             default=False,      type=str2bool,      help='Whether to use VisionAviary (default: False)', metavar='')
    parser.add_argument('--gui',                default=True,       type=str2bool,     help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=False,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=True,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=False,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--aggregate',          default=True,       type=str2bool,      help='Whether to aggregate physics steps (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=True,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=200,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=200,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=2*12,      type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--baseline',           default=True,      type=str2bool,      help='Whether to do a baseline (no learning) or use the learning algorithm', metavar='')
    parser.add_argument('--max_learn_rounds',   default=15,         type=int,           help='Maximum learning rounds for SafeOpt algorithm', metavar='')
    parser.add_argument('--ET',                 default=True,       type=str2bool,      help='Whether to use Event-trigger', metavar='')
    parser.add_argument('--beta_type',          default="const",    type=str,           help='BEta type (boguovic, const)', metavar='')
    parser.add_argument('--experiment',         default="none",      type=str,           help='Experiment kind(ATT,PWM,GE,none)', metavar='')
    parser.add_argument('--repetitions',        default=1,         type=int,           help='Number of repetitions of the experiment', metavar='')
    parser.add_argument('--lsk_plane',            default="xy",      type=str,           help='Plane for lemniscate', metavar='')
   
    ARGS = parser.parse_args()
    main()
