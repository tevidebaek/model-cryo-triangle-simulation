import numpy as np
import find_angles

import hoomd, gsd.hoomd

#here we will test out my understanding of the GSD file format

traj_file = '../Side1/trajectory.gsd'

gsd_file = gsd.hoomd.open(traj_file)

print(len(gsd_file))