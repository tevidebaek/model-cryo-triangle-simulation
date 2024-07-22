import matplotlib
import hoomd, gsd.hoomd
import numpy as np
import itertools
import math
import datetime
import os
import sys

import warnings
# matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

from create_triangle import *

import params

# here we will try to modify the code to create a system where there are three types
# of triangles

#each triangle is defined by
# a list of positions
# a list of types these are a 1 to 1 map wheren position[i] has type[i]

# Equilibrate
equilibrate = True
show_triangle = False

#interactions
int_side1 = False
int_side2 = True
int_side3 = False


# Create triangle
positions, types = get_triangle(params.triangle_length,
                                params.triangle_length,
                                params.triangle_length,
                                params.triangle_height,
                                params.angle1, params.angle2, params.angle3,
                                params.pseudoatoms_per_side,
                                params.int_atoms_per_side)

v_data = np.loadtxt('../MonomerSurfacePositions_lowdensity.csv', skiprows = 1, delimiter = ',')

positions = np.array([[v_data[i][1], v_data[i][2], v_data[i][3]] for i in range(18,len(v_data))])
interactions = np.array([[v_data[i][1], v_data[i][2], v_data[i][3]] for i in range(18)])
types = ['V' for i in range(len(positions))]
com_a = np.array([0.,0.,0.])
for i in range(len(positions)):
    com_a[0] += positions[i][0]
    com_a[1] += positions[i][1]
    com_a[2] += positions[i][2]
num_vert = np.float64(len(positions))
com_a /= num_vert
for i in range(len(positions)):
    positions[i] -= com_a
    positions[i] *= 3/1000
for i in range(len(interactions)):
    interactions[i] -= com_a
    interactions[i] *= 3/1000
#print(positions)

positions = np.append(positions, interactions,axis = 0)

#at this point, create copies of these for each hexamer body
positions_A, positions_B, positions_C = np.copy(positions), np.copy(positions), np.copy(positions)

types_A, types_B, types_C = np.copy(types), np.copy(types), np.copy(types)

#type definition for A
types_A = np.append(types_A, 'A_D1') #D1
types_A = np.append(types_A, 'A_E1') #E1
types_A = np.append(types_A, 'A_F1') #F1
types_A = np.append(types_A, 'A_C1') #C1
types_A = np.append(types_A, 'A_B1') #B1
types_A = np.append(types_A, 'A_A1') #A1
types_A = np.append(types_A, 'A_A2')
types_A = np.append(types_A, 'A_C2')
types_A = np.append(types_A, 'A_B2')
types_A = np.append(types_A, 'A_D2')
types_A = np.append(types_A, 'A_E2')
types_A = np.append(types_A, 'A_F2')
types_A = np.append(types_A, 'A_C3') #C3
types_A = np.append(types_A, 'A_D3') #D3
types_A = np.append(types_A, 'A_F3') #F3
types_A = np.append(types_A, 'A_A3') #A3
types_A = np.append(types_A, 'A_B3') #B3
types_A = np.append(types_A, 'A_E3') #E3

#type definition for B
types_B = np.append(types_B, 'B_D1') #D1
types_B = np.append(types_B, 'B_E1') #E1
types_B = np.append(types_B, 'B_F1') #F1
types_B = np.append(types_B, 'B_C1') #C1
types_B = np.append(types_B, 'B_B1') #B1
types_B = np.append(types_B, 'B_A1') #A1
types_B = np.append(types_B, 'B_A2')
types_B = np.append(types_B, 'B_C2')
types_B = np.append(types_B, 'B_B2')
types_B = np.append(types_B, 'B_D2')
types_B = np.append(types_B, 'B_E2')
types_B = np.append(types_B, 'B_F2')
types_B = np.append(types_B, 'B_C3') #C3
types_B = np.append(types_B, 'B_D3') #D3
types_B = np.append(types_B, 'B_F3') #F3
types_B = np.append(types_B, 'B_A3') #A3
types_B = np.append(types_B, 'B_B3') #B3
types_B = np.append(types_B, 'B_E3') #E3

#type definition for C
types_C = np.append(types_C, 'C_D1') #D1
types_C = np.append(types_C, 'C_E1') #E1
types_C = np.append(types_C, 'C_F1') #F1
types_C = np.append(types_C, 'C_C1') #C1
types_C = np.append(types_C, 'C_B1') #B1
types_C = np.append(types_C, 'C_A1') #A1
types_C = np.append(types_C, 'C_A2')
types_C = np.append(types_C, 'C_C2')
types_C = np.append(types_C, 'C_B2')
types_C = np.append(types_C, 'C_D2')
types_C = np.append(types_C, 'C_E2')
types_C = np.append(types_C, 'C_F2')
types_C = np.append(types_C, 'C_C3') #C3
types_C = np.append(types_C, 'C_D3') #D3
types_C = np.append(types_C, 'C_F3') #F3
types_C = np.append(types_C, 'C_A3') #A3
types_C = np.append(types_C, 'C_B3') #B3
types_C = np.append(types_C, 'C_E3') #E3

if len(positions) != len(types):
    print('Wrong number of positions and types!!!')

if show_triangle:
    xcoords = [positions[i][0] for i in range(len(positions))]
    ycoords = [positions[i][1] for i in range(len(positions))]
    zcoords = [positions[i][2] for i in range(len(positions))]
    plotTriangle(xcoords,ycoords,zcoords, types)
    plt.show()
    quit()

#the following computation will be true for all the three types
#compute the inertia tensor for the pentagonal subunit
#as a sum over each vertex particle

#convert list of positions to nparray for calculation
r   = np.array(positions, dtype=float)

#compute the first contribution to the inertia tensor
moi = (np.dot(r[0], r[0]) * np.identity(3) - np.outer(r[0], r[0]))

#loop over other particles - just the vertices
for i in range(1,3):
    moi += (np.dot(r[i], r[i]) * np.identity(3) - np.outer(r[i], r[i]))

#print the inertia tensor to confirm it is diagonal
print('inertia tensor ...')
print(moi) #xy component may be nonzero, but this is numerical roundoff error

#moi is already diagonal, extract these components
Ixx = moi[0][0]
Iyy = moi[1][1]
Izz = moi[2][2]
inertia_tensor = [Ixx, Iyy, Izz]

orientations = [(1,0,0,0)] * len(positions)

rigid = hoomd.md.constrain.Rigid()

#Rigid body definition for A
name_A = "Triangle_A"
rigid.body[name_A] = {
     "constituent_types": types_A,
     "positions":         positions_A,
     "orientations":      orientations
}

rigid_eq = hoomd.md.constrain.Rigid()
rigid_eq.body[name_A] = {
     "constituent_types": types_A,
     "positions":         positions_A,
     "orientations":      orientations
}

#Rigid body definition for B
name_B = "Triangle_B"
rigid.body[name_B] = {
     "constituent_types": types_B,
     "positions":         positions_B,
     "orientations":      orientations
}

#rigid_eq = hoomd.md.constrain.Rigid()  #dont do this again
rigid_eq.body[name_B] = {
     "constituent_types": types_B,
     "positions":         positions_B,
     "orientations":      orientations
}

#Rigid body definition for C
name_C = "Triangle_C"
rigid.body[name_C] = {
     "constituent_types": types_C,
     "positions":         positions_C,
     "orientations":      orientations
}

#rigid_eq = hoomd.md.constrain.Rigid()  #dont do this again
rigid_eq.body[name_C] = {
     "constituent_types": types_C,
     "positions":         positions_C,
     "orientations":      orientations
}

full_type_list = [name_A] + [name_B] + [name_C] + list(set(types_A)) + list(set(types_B)) + list(set(types_C)) 

snapshot = gsd.hoomd.Frame()
snapshot.particles.N           = params.subunits                                 #number of subunit
snapshot.configuration.box     = [params.box_size, params.box_size, params.box_size, 0, 0, 0]   #square box with set size
snapshot.particles.orientation = [(1,0,0,0)] * params.subunits                   #orientations set to default
snapshot.particles.types       = full_type_list          #[rigid body name, particle types]
snapshot.particles.typeid      = [0,1,2]*int(params.subunits/3)                        #index in types of each subunit, make sure to use typeids of the rigid bodies
snapshot.particles.moment_inertia = inertia_tensor * params.subunits             #inertia tensor of each subunit

print("snapshot types")
print(snapshot.particles.types)

print("snapshot type ids")
print(snapshot.particles.typeid)

#take cube root of number of subunits to get how many in each dimension - round up to overcount
particle_per_dim = int(np.ceil(params.subunits ** (1/3)))

#init storage for positions, each subunit needs x,y,z coordinates
positions = np.zeros((params.subunits, 3), dtype=float)

#discretize an interval with equally spaced points on [-box_size/2, box_size/2]
x1 = np.linspace(-params.box_size / 2.0, params.box_size / 2, particle_per_dim+1)[:-1]

#use coordinates in x1 to generate points in all 3 dimensions
c  = 0
for i in range(particle_per_dim):
    for j in range(particle_per_dim):
        for k in range(particle_per_dim):


            positions[c][0] = x1[i]
            positions[c][1] = x1[j]
            positions[c][2] = x1[k]
            c+=1

            if c >= params.subunits:
                break

#visualize the lattice
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='red')
# ax.view_init(30, -120)
# plt.show()


#add the lattice positions to the snapshot
snapshot.particles.position = positions

#save the snapshot as a gsd file
if not os.path.isfile('lattice_3d.gsd'):
    with gsd.hoomd.open(name='lattice_3d.gsd', mode='w') as f:
        f.append(snapshot)
        f.close()
else:
    print('Lattice file lattice_3d.gsd already exists')

#set the simulation device
#my_device_eq = hoomd.device.GPU() #If you have a GPU available, change CPU() to GPU() for large speedup!
my_device_eq = hoomd.device.CPU() #If you have a GPU available, change CPU() to GPU() for large speedup!

#create a simulation object to manage all the simulation setup
simulation_eq = hoomd.Simulation(device=my_device_eq, seed=params.seed)

#create an integrator to update the subunits every timestep
integrator_eq = hoomd.md.Integrator(dt=params.dt, integrate_rotational_dof=True)
integrator_eq.rigid = rigid_eq

#filter which rigid bodies this integrator manages, and set integrator type
filtered_bodies = hoomd.filter.Rigid(("center","free"))
L_int_eq = hoomd.md.methods.Langevin(filter=filtered_bodies, kT=params.kT)
integrator_eq.methods.append(L_int_eq)
simulation_eq.operations.integrator = integrator_eq

#create a neignborlist for potential calculations. This speeds up pairwise distance checks
cell = hoomd.md.nlist.Cell(buffer=3.0, exclusions=['body']) #particle in the same rigid body do not need to be checked

#initialize a Lennard-Jones potential as the repulsive equilibration potential
lj_eq = hoomd.md.pair.LJ(nlist=cell)

#set an effective repulsive distance for the subunits, and a cutoff for a WCA potential
r_dist = 1.25 * 2.1
r_cut  = r_dist * 2.0 ** (1.0/6.0)

#set a repulsive interaction between each pair of atom types, with strength 1
for type1 in full_type_list:
    for type2 in full_type_list:

        lj_eq.params[(type1,type2)] = dict(epsilon=1, sigma=r_dist)
        lj_eq.r_cut[ (type1,type2)] = r_cut

#add the potential to the integrator
integrator_eq.forces.append(lj_eq)

eq_steps = 20000 #number of equilibration time steps

#make a logger that outputs the frame at the final frame of equilibration
gsd_writer_eq = hoomd.write.GSD(filename='equilibrated_start_3d.gsd',
                                trigger=hoomd.trigger.Periodic(params.t_dump),
                                mode='xb',
                                filter=hoomd.filter.All())
#append the logger to the simulation object
simulation_eq.operations.writers.append(gsd_writer_eq)

#make a logger to track progress and efficiency of simulation
log_period = int(params.t_log)
logger_eq  = hoomd.logging.Logger(categories=['scalar'])
logger_eq.add(simulation_eq, quantities=['timestep','tps'])
sim_writer = hoomd.write.Table(trigger=hoomd.trigger.Periodic(log_period),
                               logger=logger_eq)
simulation_eq.operations.writers.append(sim_writer)

#create the simulation state from the lattice. add the rigid bodies
simulation_eq.create_state_from_gsd(filename='lattice_3d.gsd')
rigid.create_bodies(simulation_eq.state)

if equilibrate:
    print('equilibrating...')
    #run the equilibration
    simulation_eq.run(eq_steps)
    gsd_writer_eq.flush()

#device = hoomd.device.GPU()
device = hoomd.device.CPU()

simulation = hoomd.Simulation(device=device, seed=params.seed)
simulation.timestep = 0

simulation.create_state_from_gsd('equilibrated_start_3d.gsd', frame= -1)

integrator = hoomd.md.Integrator(dt = params.dt, integrate_rotational_dof = True)
integrator.rigid = rigid
simulation.operations.integrator = integrator

filtered_bodies = hoomd.filter.Rigid(("center", "free"))

#use this line for a temperature ramp
#L_int = hoomd.md.methods.Langevin(filter = filtered_bodies, kT=hoomd.variant.Ramp(A = 1.0, B = 0.0, t_start = 0, t_ramp = 100_000_000))
#integrator.methods.append(L_int)

#use this line for a constant temperature
L_int = hoomd.md.methods.Langevin(filter = filtered_bodies, kT=0.7)
integrator.methods.append(L_int)

bussi = hoomd.md.methods.thermostats.Bussi(1.0)
bussi.kT = hoomd.variant.Ramp(A = 1.0, B = 0.0, t_start = 0, t_ramp = 100_000_000)
# simulation.operations.integrator.methods[0].thermostat = bussi

cell = hoomd.md.nlist.Cell(buffer=3.0, exclusions=['body'])

lj    = hoomd.md.pair.LJ(   nlist = cell, default_r_cut=3.0, mode='shift')
morse = hoomd.md.pair.Morse(nlist = cell, default_r_cut=3.0, mode='shift')

#init all the interactions to 0
lj.params[(full_type_list, full_type_list)]     = dict(epsilon=0, sigma=0)
lj.r_cut[  (full_type_list, full_type_list)]    = 0
morse.params[(full_type_list, full_type_list)]  = dict(D0=0, r0=0, alpha=0)
morse.r_cut[  (full_type_list, full_type_list)] = 0

# Repulsive interaction between V and S
lj.params[("V", "V") ] = dict(epsilon = params.E_rep, sigma = params.lj_sigma)
lj.r_cut[ ("V", "V") ] = params.lj_cut
# lj.params[("V", "S") ] = dict(epsilon = params.E_rep, sigma = params.lj_sigma)
# lj.r_cut[ ("V", "S") ] = params.lj_cut
# lj.params[("S", "S") ] = dict(epsilon = params.E_rep, sigma = params.lj_sigma)
# lj.r_cut[ ("S", "S") ] = params.lj_cut

integrator.forces.append(lj)

#new interactions with the 6 binding sites per side

#side 1 interaction
morse.params[ ("A_A1", "B_F1") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("A_A1", "B_F1") ] = params.m_cut
morse.params[ ("A_B1", "B_E1") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("A_B1", "B_E1") ] = params.m_cut
morse.params[ ("A_C1", "B_D1") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("A_C1", "B_D1") ] = params.m_cut

#side 2 interactions
morse.params[ ("B_A2", "C_F2") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("B_A2", "C_F2") ] = params.m_cut
morse.params[ ("B_B2", "C_E2") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("B_B2", "C_E2") ] = params.m_cut
morse.params[ ("B_C2", "C_D2") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("B_C2", "C_D2") ] = params.m_cut

#side 3 interactions
morse.params[ ("C_A3", "A_F3") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("C_A3", "A_F3") ] = params.m_cut
morse.params[ ("C_B3", "A_E3") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("C_B3", "A_E3") ] = params.m_cut
morse.params[ ("C_C3", "A_D3") ] = dict(D0=params.E_bond, r0=params.m_r0, alpha=params.m_alpha)
morse.r_cut[  ("C_C3", "A_D3") ] = params.m_cut

integrator.forces.append(morse)

#define a writer for the gsd file and add it to simulation
gsd_writer = hoomd.write.GSD(filename="trajectory.gsd",
                             trigger=hoomd.trigger.Periodic(params.t_dump),
                             mode='xb')
simulation.operations.writers.append(gsd_writer)

#define a logger/writer for the simulation details
logger = hoomd.logging.Logger(categories=['scalar','string'])
logger.add(simulation, quantities=['timestep', 'tps', 'walltime'])


#make a table of quantities to log and add it to sim
sim_writer = hoomd.write.Table(trigger=hoomd.trigger.Periodic(params.t_log),
                               logger = logger)
simulation.operations.writers.append(sim_writer)

t_max = 10000

print('running the simulation...')

#run the simulation
for t_partial in range(t_max):
    print(str(t_partial), '/', str(t_max))

    print(simulation.operations.integrator.methods[0].kT.__getstate__())
    simulation.run(params.num_steps//t_max)
    gsd_writer.flush()
