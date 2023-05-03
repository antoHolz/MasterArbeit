import numpy as np
import scipy.interpolate as si
import os
import sys

def beta_bog(learn_counter):
    return 0.8*np.log(2*learn_counter)   #bogunovic 0.8*np.log(2*learn_counter)
def beta_const(learn_counter):
    return 2

def get_trajectory_xy(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + start_pos[0], R * np.sin(
            2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2 + start_pos[1],  start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    return TARGET_POS

def bern_lemniscate_xy(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = 0.75*R * np.sqrt(2)* np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2)/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)
            **2+1) + start_pos[0],  R * np.sqrt(2)* np.cos((i / NUM_WP) * (2 * np.pi) +
             np.pi / 2)* np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2 )/(np.sin((i / NUM_WP)
              * (2 * np.pi) + np.pi / 2)**2+1) + start_pos[1], start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    return TARGET_POS

def eck_lemniscate_xy(NUM_WP, NUM_R, start_pos, width, height):
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        if(i>5*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = width-width*(i-5*NUM_WP/6)/(NUM_WP/6)+ start_pos[0],height -height*(i-5*NUM_WP/6)/(NUM_WP/6)+ start_pos[1],start_pos[2]
        elif(i>4*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] =  width+ start_pos[0], -height + 2*height*(i-4*NUM_WP/6)/(NUM_WP/6)+ start_pos[1], start_pos[2]
        elif(i>2*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = -width + width*(i-2*NUM_WP/6)/(NUM_WP/6)+ start_pos[0], height - height*(i-2*NUM_WP/6)/(NUM_WP/6)+ start_pos[1], start_pos[2]
        elif(i>1*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = -width+ start_pos[0],-height+ 2*height*(i-NUM_WP/6)/(NUM_WP/6)+ start_pos[1], start_pos[2]
        else:
            TARGET_POS[i+NUM_R, :] = -width*i/(NUM_WP/6)+ start_pos[0],-height*i/(NUM_WP/6)+ start_pos[1],start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
   #TARGET_POS[abs(TARGET_POS)<1e-5]=0
    return TARGET_POS

def get_trajectory_xz(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = R * np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2) + start_pos[0], start_pos[1], R * np.sin(
            2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+  start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    return TARGET_POS

def bern_lemniscate(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = 0.75*R * np.sqrt(2)* np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2)/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)
            **2+1) + start_pos[0], start_pos[1], R * np.sqrt(2)* np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2)* np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2
            )/(np.sin((i / NUM_WP) * (2 * np.pi) + np.pi / 2)**2+1) + start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    return TARGET_POS

def eck_lemniscate(NUM_WP, NUM_R, start_pos, width, height):
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        if(i>5*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = width-width*(i-5*NUM_WP/6)/(NUM_WP/6)+ start_pos[0],start_pos[1],height -height*(i-5*NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        elif(i>4*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] =  width+ start_pos[0], start_pos[1], -height + 2*height*(i-4*NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        elif(i>2*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = -width + width*(i-2*NUM_WP/6)/(NUM_WP/6)+ start_pos[0], start_pos[1], height - height*(i-2*NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        elif(i>1*NUM_WP/6):
            TARGET_POS[i+NUM_R, :] = -width+ start_pos[0], start_pos[1],-height+ 2*height*(i-NUM_WP/6)/(NUM_WP/6)+ start_pos[2]
        else:
            TARGET_POS[i+NUM_R, :] = -width*i/(NUM_WP/6)+ start_pos[0],start_pos[1],-height*i/(NUM_WP/6)+ start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
   
    return TARGET_POS

def flip(fliptime, returntime, ctrl_freq, hoverheight):
    ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
    ### Trajectory
    pos_ref_points = np.loadtxt(ROOT_DIR+"/pos_ref_flip.csv", delimiter=',')
    pos_ref_time = np.linspace(0, fliptime, pos_ref_points.shape[0])
    flip_traj = [si.splrep(pos_ref_time, pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, 0*pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, pos_ref_points[:, 1], k=3)]
    quat_points = -1.99999 / (1 + np.exp(-20*(pos_ref_time-0.9/2))) + 1.99999/2
    rot_traj = si.splrep(pos_ref_time, quat_points, k=3)
    
    ts=np.linspace(0, fliptime, int(fliptime*ctrl_freq))
    target_pos = np.array([si.splev(ts, flip_traj[i]) for i in range(3)]).T
    target_pos[:,2] = target_pos[:,2] + 1
    target_vel = np.array([si.splev(ts, flip_traj[i], der=1) for i in range(3)]).T
    target_acc = np.array([si.splev(ts, flip_traj[i], der=2) for i in range(3)]).T
    q0 = si.splev(ts, rot_traj)
    q2 = np.sqrt(1 - q0**2)
    target_quat = np.zeros((len(q0),4)) #np.array([q0, 0, q2, 0])
    target_quat[:,3]=q0
    target_quat[:,1]=q2
    dq0 = si.splev(ts, rot_traj, der=1)
    dq2 = - dq0 * q0 / q2
    target_quat_vel = np.zeros((len(dq0),4))
    target_quat_vel[:,3]=dq0
    target_quat_vel[:,1]=dq2
    target_ang_vel = np.roll(np.array([(2 * quat_mult(quat_conj(target_quat[i,:]), target_quat_vel[i,:]))[0:3] for i in range(len(q0))]),-1) #check quat
    #np.array([(2 * quat_mult(quat_conj(target_quat[i,:]), target_quat_vel[i,:]))[0:3] for i in range(len(q0))])
    return target_pos, target_vel, target_acc, target_quat, target_ang_vel


def get_flip_trajectory(fliptime, returntime, ctrl_freq, hoverheight):
    ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
    ### Trajectory
    pos_ref_points = np.loadtxt(ROOT_DIR+"/pos_ref_flip.csv", delimiter=',')
    pos_ref_time = np.linspace(0, fliptime, pos_ref_points.shape[0])
    flip_traj = [si.splrep(pos_ref_time, pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, 0*pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, pos_ref_points[:, 1], k=3)]
    quat_points = -1.99999 / (1 + np.exp(-20*(pos_ref_time-0.9/2))) + 1.99999/2
    rot_traj = si.splrep(pos_ref_time, quat_points, k=3)

    target_pos=np.zeros((int((fliptime+returntime)*ctrl_freq), 3))
    target_pos[:,2]=hoverheight
    target_vel=np.zeros((int((fliptime+returntime)*ctrl_freq), 3))
    target_acc=np.zeros((int((fliptime+returntime)*ctrl_freq), 3))
    target_ang_vel=np.zeros((int((fliptime+returntime)*ctrl_freq), 3))
    target_quat = np.zeros((int((fliptime+returntime)*ctrl_freq), 4))
    target_quat_vel = np.zeros((int((fliptime+returntime)*ctrl_freq), 4))
    
    ts=np.linspace(0, fliptime, int(fliptime*ctrl_freq))
    target_pos[int((returntime)*ctrl_freq):,:] = np.array([si.splev(ts, flip_traj[i]) for i in range(3)]).T
    target_pos[int((returntime)*ctrl_freq):,2] = target_pos[int((returntime)*ctrl_freq):,2] + hoverheight
    target_vel[int((returntime)*ctrl_freq):,:] = np.array([si.splev(ts, flip_traj[i], der=1) for i in range(3)]).T
    target_acc[int((returntime)*ctrl_freq):,:] = np.array([si.splev(ts, flip_traj[i], der=2) for i in range(3)]).T
    q0 = si.splev(ts, rot_traj)
    q2 = np.sqrt(1 - q0**2)
    #target_quat = np.zeros((len(q0),4)) #np.array([q0, 0, q2, 0])
    target_quat[int((returntime)*ctrl_freq):,3]=q0
    target_quat[int((returntime)*ctrl_freq):,1]=q2
    dq0 = si.splev(ts, rot_traj, der=1)
    dq2 = - dq0 * q0 / q2
    #target_quat_vel = np.zeros((len(dq0),4))
    target_quat_vel[int((returntime)*ctrl_freq):,3]=dq0
    target_quat_vel[int((returntime)*ctrl_freq):,1]=dq2
    target_ang_vel[int((returntime)*ctrl_freq):,:] = np.roll(np.array([(2 * quat_mult(quat_conj(target_quat[i,:]), target_quat_vel[i,:]))[0:3] for i in range(len(q0))]),-1) #check quat
    #np.array([(2 * quat_mult(quat_conj(target_quat[i,:]), target_quat_vel[i,:]))[0:3] for i in range(len(q0))]    
    return target_pos, target_vel, target_acc, target_quat, target_ang_vel

def get_trajectory_xyf(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + start_pos[0], R * np.sin(
            2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2 + start_pos[1],  start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    TARGET_VEL = np.zeros((NUM_WP+NUM_R, 3)) #Actualize!
    TARGET_ACC = np.zeros((NUM_WP+NUM_R, 3))
    TARGET_QUAT = np.zeros((NUM_WP+NUM_R, 4))
    TARGET_ANG_VEL = np.zeros((NUM_WP+NUM_R, 3))
    return TARGET_POS, TARGET_VEL, TARGET_ACC, TARGET_QUAT, TARGET_ANG_VEL

def get_trajectory_xzf(NUM_WP,NUM_R, start_pos= np.array([0, 0,1.]), R=.3):
    #Lemniskate von Gerono
    TARGET_POS = np.zeros((NUM_WP+NUM_R, 3))
    for i in range(NUM_WP):
        TARGET_POS[i+NUM_R, :] = R * np.cos((i / NUM_WP) * (
            2 * np.pi) + np.pi / 2) + start_pos[0], start_pos[1], R * np.sin(
            2*((i / NUM_WP) * (2 * np.pi) + np.pi / 2))/2+  start_pos[2]
    for i in range(NUM_R):
        TARGET_POS[i, :] = start_pos
    TARGET_VEL = np.zeros((NUM_WP+NUM_R, 3)) #actualize!!
    TARGET_ACC = np.zeros((NUM_WP+NUM_R, 3))
    TARGET_QUAT = np.zeros((NUM_WP+NUM_R, 4))
    TARGET_ANG_VEL = np.zeros((NUM_WP+NUM_R, 3))
    return TARGET_POS, TARGET_VEL, TARGET_ACC, TARGET_QUAT, TARGET_ANG_VEL

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")

def quat_mult(quaternion1, quaternion0):
    """Multiply two quaternions in scalar last form"""
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y0 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def quat_conj(quat):
    """Return conjugate of a quaternion in scalar last form"""
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]])