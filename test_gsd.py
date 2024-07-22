import numpy as np
import find_angles

import hoomd, gsd.hoomd

#here we will test out my understanding of the GSD file format

traj_file = './SimulationOutput/Side1/trajectory.gsd'

gsd_file = gsd.hoomd.open(traj_file)

print(len(gsd_file))

#try to look at the index of partciles with type "A1", which is an interaction on side 1
A_index = np.where(np.array(snap.particles.types) == "A1")[0]

print(A_index)
