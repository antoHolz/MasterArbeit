import numpy as np
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
import xml.etree.ElementTree as etxml
import os
import pybullet as p

class GeomControl(BaseControl):
    """
    Geometric control class for quadcopters.
    """

    ################################################################################

    def __init__(self, 
                drone_model: DroneModel,
                 g: float=9.81,
                 drone_type='cf2',
                 dt=0.005
                 ):
        super().__init__(drone_model=drone_model, g=g)

        if (drone_type == 'cf2'):
            self.k_r = np.array([0.5, 0.5, 1.25]) #4.5#
            self.k_v = np.array([0.2, 0.2, 0.8]) #0.3#
            self.k_R = 8e-2*np.ones(3) #0.2
            self.k_w = 2e-3*np.ones(3) 
            self.inertia = np.diag([1.4e-5, 1.4e-5, 2.1e-5])
            self.mass = 0.028
        elif (drone_type == 'large_quad'):
            self.k_r = 6
            self.k_v = 3
            self.k_R = 8
            self.k_w = 0.2
            self.inertia = np.diag([1.5e-3, 1.45e-3, 2.66e-3])
            self.mass = 0.407

        else:
            raise NotImplementedError

        self.gravity = 9.81

        # optional integrator term
        self.dt = dt
        self.int_pos_e = np.zeros(3)

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])

        self.l = 0.046/np.sqrt(2)
        self.k = 3.16e-10*(60/2/np.pi)**2
        self.b = 7.94e-12*(60/2/np.pi)**2
        self.INP = np.array([[1, 1, 1, 1], [self.l, self.l, -self.l, -self.l], [-self.l, self.l, self.l, -self.l], [-self.b/self.k, self.b/self.k, -self.b/self.k, self.b/self.k]])
        # self.thrust_data = np.loadtxt('../files/thrust.csv', dtype=float, delimiter=',')
        # self.pos_data = np.loadtxt('../files/pos.csv', dtype=float, delimiter=',')
        self.RPMfactor=1
        self.reset()
        #np.array([[1, 1, 1, 1], [-self.l, -self.l, self.l, self.l], [self.l, -self.l, -self.l, self.l],[self.b/self.k, -self.b/self.k, self.b/self.k, -self.b/self.k]])

    ################################################################################

    def compute_pos_control(self,
                            cur_pos,
                            cur_quat,
                            cur_vel,
                            cur_ang_vel,
                            target_pos,
                            target_rpy=np.zeros(3),
                            target_vel=np.zeros(3),
                            target_rpy_rates=np.zeros(3)
                            ):
        """
        Geometric ctrl of quadcopters: position ctrl based on reference position and velocity.
        """
        p_target = target_pos
        pos_e = cur_pos - target_pos
        vel_e = cur_vel - target_vel
        target_acc= np.array([0, 0, 0])
        target_yaw = target_rpy[2]
        self.int_pos_e = self.int_pos_e + self.dt * pos_e

        A = -self.k_r * pos_e - self.k_v * vel_e +\
             self.mass * self.gravity * np.array([0, 0, 1]) + self.mass * target_acc
        r3 = A / np.linalg.norm(A)
        #In newer code with some other stuff
        if np.abs(target_yaw) < 1e-3:  # speed up cross product if yaw target is zero
            cross_temp = self._my_cross(r3)
            r2 = cross_temp / np.linalg.norm(cross_temp)
            r1 = self._my_cross_2(r2, r3)
        else:
            cross_temp = np.cross(r3, np.array([np.cos(target_yaw), np.sin(target_yaw), 0]))
            r2 = cross_temp / np.linalg.norm(cross_temp)
            r1 = np.cross(r2, r3)
        target_rotation = np.array([r1, r2, r3]).transpose()
        #cur_quat = np.roll(cur_quat, -1) This causes issues!!! Probably pybullet already gives scalar last format
        cur_rotation = Rotation.from_quat(cur_quat).as_matrix()

        rot_e =  1 / 2 * self._veemap(np.dot((target_rotation.transpose()), cur_rotation)
                                     - np.dot(cur_rotation.transpose(), target_rotation))
        if np.isnan(np.sum(rot_e)):
            rot_e = np.zeros(3)
        ang_vel_e = cur_ang_vel - np.dot(np.dot(cur_rotation.transpose(), target_rotation), target_rpy_rates)
        eta_R, mu_R = self._mu_R(cur_quat, cur_ang_vel, rot_e, ang_vel_e)
        target_torques = -self.k_R * rot_e - self.k_w * ang_vel_e + np.cross(cur_ang_vel,
                         np.dot(self.inertia, cur_ang_vel)) + eta_R + mu_R

        thrust = np.dot(A, np.dot(cur_rotation, np.array([0, 0, 1])))

        ctrl = np.zeros(4)
        ctrl[0] = thrust
        ctrl[1:4] = target_torques

        f = np.dot(np.linalg.inv(self.INP), ctrl)
        f = np.clip(f, 0, 0.16)
        rpm = np.sqrt(f/self.k)*30/np.pi
        # rpm = np.clip(rpm, 0, 21666)

        actual_torque = np.dot(self.INP, self.k*(rpm*np.pi/30)**2)[1:4]

        target_euler = np.array([0, 0, 0])  # (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)

        cur_rpy = p.getEulerFromQuaternion(cur_quat)


        return rpm

        # return target_torques


    def compute_att_control(self,
                            cur_pos,
                            cur_quat,
                            cur_vel,
                            cur_ang_vel,
                            target_pos,
                            target_vel,
                            target_acc,
                            target_quat,
                            target_ang_vel,
                            ):
        """
        Geometric ctrl of quadcopters: attitude ctrl based on reference quaternion, quaternion derivative,
        position and velocity. Quaternion should be in scalar first format.
        """
        # # Convert quaternion from scalar first to scalar last format (not necessary in pybullet!!)
        # cur_quat = np.roll(cur_quat, -1) 
        # target_quat = np.roll(target_quat, -1)

        # omega = Im{2*conj(q)*dq}
        # target_ang_vel = (2 * self._quat_mult(self._quat_conj(target_quat), target_quat_vel))[0:3]
        target_rotation = Rotation.from_quat(target_quat).as_matrix()
        cur_rotation = Rotation.from_quat(cur_quat).as_matrix()
        rot_e = 1 / 2 * self._veemap(np.dot((target_rotation.transpose()), cur_rotation)
                                     - np.dot(cur_rotation.transpose(), target_rotation))
        ang_vel_e = cur_ang_vel - np.dot(np.dot(cur_rotation.transpose(), target_rotation), target_ang_vel)
        eta_R, mu_R = self._mu_R(cur_quat, cur_ang_vel, rot_e, ang_vel_e)
        target_torques = -self.k_R * rot_e - self.k_w * ang_vel_e + \
                         np.cross(cur_ang_vel, np.dot(self.inertia, cur_ang_vel)) + eta_R + mu_R
        pos_e = cur_pos - target_pos
        vel_e = cur_vel - target_vel
        e3 = np.array([0, 0, 1])
        den = np.dot(e3, np.dot(cur_rotation, e3))
        self.int_pos_e = self.int_pos_e + self.dt * pos_e
        if np.abs(den) > 1e-7:
            thrust = (-self.k_r[2] * pos_e[2] - self.k_v[2] * vel_e[2] + self.mass * self.gravity +
                      self.mass * target_acc[2]) / den
        else:
            print('den is close to zero')
            thrust = 0

        ctrl = np.zeros(4)
        ctrl[0] = thrust
        ctrl[1:4] = target_torques
        f = np.dot(np.linalg.inv(self.INP), ctrl)
        f = np.clip(f, 0, 0.16)
        rpm = np.sqrt(f/self.k)*30/np.pi*self.RPMfactor

        return rpm

################################################################################

    def setCtrlCoefficients(self,
                           K_rx=None,
                           K_rz=None,
                           K_vx=None,
                           K_vz=None,
                           K_R=None,
                           K_w=None,
                           ):
        """Sets the coefficients of a controller.

        This method throws an error message and exist is the coefficients
        were not initialized (e.g. when the controller is not a PID one).

        Parameters
        ----------
        k_r : #ndarray, optional
            #(3,1)-shaped array of floats containing the position control proportional coefficients.

        """
        ATTR_LIST = ['k_r', 'k_v', 'k_R', 'k_w']
        if not all(hasattr(self, attr) for attr in ATTR_LIST):
            print("[ERROR] in BaseControl.setPIDCoefficients(), not all PID coefficients exist as attributes in the instantiated control class.")
            exit()
        else:
            self.k_r[0:2] = self.k_r[0:2] if K_rx is None else K_rx
            self.k_r[2] = self.k_r[2] if K_rz is None else K_rz
            self.k_v[0:2] = self.k_v[0:2] if K_vx is None else K_vx
            self.k_v[2] = self.k_v[2] if K_vz is None else K_vz
            self.k_R = self.k_R if K_R is None else K_R*np.ones(3)
            self.k_w = self.k_w if K_w is None else K_w*np.ones(3)

    def getCtrlCoefficients(self):
        return self.k_r, self.k_v, self.k_w, self.k_R

    def setRPMfactor(self,factor):
        self.RPMfactor=factor

    def setMass(self,mass):
        self.mass=mass

    def _mu_r(self, pos_e, vel_e):
        return np.zeros(3), np.zeros(3)

    def _mu_R(self, cur_quat, cur_ang_vel, rot_e, ang_vel_e):
        return np.zeros(3), np.zeros(3)

    @staticmethod
    def _veemap(mat):
        a = np.zeros(3)
        a[0] = mat[2, 1]
        a[1] = mat[0, 2]
        a[2] = mat[1, 0]
        return a

    @staticmethod
    def _hatmap(a):
        mat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return mat

    @staticmethod
    def _quat_mult(quaternion1, quaternion0):
        """Multiply two quaternions in scalar last form"""
        x0, y0, z0, w0 = quaternion0
        x1, y1, z1, w1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y0 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    @staticmethod
    def _quat_conj(quat):
        """Return conjugate of a quaternion in scalar last form"""
        return np.array([-quat[0], -quat[1], -quat[2], quat[3]])

    def stability_analysis(self, k_r, k_v, k_R, k_w, c1, c2, J, m, eps=None):

        ######## Stability analysis of geometric ctrl based on the original paper #################

        Lmax = max(np.linalg.eigvals(J))
        Lmin = min(np.linalg.eigvals(J))
        psi1 = 0.01
        alpha = np.sqrt(psi1 * (2 - psi1))
        e_rmax = 0.001  # max{||e_v(0)||, B/(k_v*(1-alpha))}, TODO: see (23), (24)
        B = 0.0001  # B > ||-mge_3 + m\ddot{x}||  TODO: calculate B, see (16)
        W1 = np.array([[c1 * k_r / m * (1 - alpha), -c1 * k_v / (2 * m) * (1 + alpha)],
                       [-c1 * k_v / (2 * m) * (1 + alpha), k_v * (1 - alpha) - c1]])
        W12 = np.array([[c1 / m * B, 0], [B + k_r * e_rmax, 0]])
        W2 = np.array([[c2 * k_R / Lmax, -c2 * k_w / (2 * Lmin)], [-c2 * k_w / (2 * Lmin), k_w - c2]])
        exp1 = min([k_v * (1 - alpha),
                    4 * m * k_r * k_v * (1 - alpha) ** 2 / (k_v ** 2 * (1 + alpha) ** 2 + 4 * m * k_r * (1 - alpha)),
                    np.sqrt(k_r * m)])
        crit1 = c1 < exp1
        exp2 = min([k_w, 4 * k_w * k_R * Lmin ** 2 / (k_w ** 2 * Lmax + 4 * k_R * Lmin ** 2), np.sqrt(k_R * Lmin)])
        crit2 = c2 < exp2
        exp3 = [min(np.linalg.eigvals(W2)), 4 * np.linalg.norm(W12, ord=2) ** 2 / min(np.linalg.eigvals(W1))]
        crit3 = exp3[0] > exp3[1]

        return crit1, crit2, crit3

    @staticmethod
    def _my_cross(r3):
        '''
        Simplify cross product if the second vector is [0, 0, 1].
        '''
        return np.array([0, r3[2], -r3[1]])

    @staticmethod
    def _my_cross_2(a, b):
        '''
        Simplify cross product if the first vector is [0, a2, a3].
        '''
        return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0], -a[1]*b[0]])
