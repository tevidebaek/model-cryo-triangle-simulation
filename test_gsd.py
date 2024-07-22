import numpy as np
import find_angles

import hoomd, gsd.hoomd

#here we will test out my understanding of the GSD file format

traj_file = './SimulationOutput/Side1/trajectory.gsd'

gsd_file = gsd.hoomd.open(traj_file)

print(len(gsd_file))

#create a snapshot of a single frame of the trajectory
snap = gsd_file[100]   #100 is a random frame
print('available particle types ', snap.particles.types)

#try to look at the index of partciles with type "A1", which is an interaction on side 1
#this just pulls the index of the type in the list of types, not the index of particles with this type
A_index = np.where(np.array(snap.particles.types) == "A1")[0][0]
F_index = np.where(np.array(snap.particles.types) == "F1")[0][0]

print('particle list ', snap.particles.position)
