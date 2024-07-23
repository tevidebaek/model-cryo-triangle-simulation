import numpy as np # only for np.pi
# Parameters
triangle_length = 2.
triangle_height = triangle_length/3.
# angles for m=10, n=0
angle1 = 70 *np.pi/180
angle2 = 100*np.pi/180
angle3 = 70 *np.pi/180
# angles for m=10, n=2
angle1 = 76.65 *np.pi/180
angle2 = 98.516 * np.pi/180
angle3 = 66.858 * np.pi/180
pseudoatoms_per_side = 6
int_atoms_per_side = 3

kT = 1
dt = 0.002
num_steps = 1000000
t_dump = 2000
t_log = 20000
subunits = 36*6
box_size = 30
seed = 123

# 15 to 20 per side
E_bond = 8.0
E_rep = E_bond / 6.0

# LJ repulsion
lj_sigma = 0.5 / triangle_length  #was 0.8
lj_cut = lj_sigma * 2**(1./6.)

# Morse attraction
m_r0 = 0.3 / triangle_length   #was 0.7
m_alpha = 3. * triangle_length
m_cut = 4./3. / triangle_length

