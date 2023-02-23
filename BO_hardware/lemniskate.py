#!/usr/bin/env python

import csv
import numpy as np
from pycrazyswarm import *

Z = 1.


def get_postion(phi):
    x = 2*np.cos(phi)
    y = 2*np.sin(2*phi) / 2
    return np.array([x, y, Z])


def deg2rad(angle):
    return angle * np.pi / 180.


if __name__ == "__main__":
    csv_file_name = "tajectories.csv"  # tajectories_wind

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
        wr.writerow(["ref_x", "ref_y", "ref_z", "x", "y", "z"])

    for i, t in enumerate(range(90, 360 + 90)):

        for cf in allcfs.crazyflies:
            new_angle = deg2rad(t)
            ref = get_postion(new_angle)
            cf.cmdPosition(ref, 0,)
            position = list(np.append(ref, cf.position()))

            with open(csv_file_name, 'a', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow(position)

        timeHelper.sleep(0.1)

    allcfs.land(targetHeight=0.05, duration=2.0)
