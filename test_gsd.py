import numpy as np
import find_angles

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hoomd, gsd.hoomd

#############################################################
#                       MATH FUNCTIONS                      #
#############################################################

def deg2rad(ang):
    ang = ang*np.pi/180
    return ang

def dot(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def norm(a):
  return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def cross(a,b):
    return np.array([a[1]*b[2] - b[1]*a[2], a[2]*b[0] - b[2]*a[0], a[0]*b[1] - b[0]*a[1]])

def angleBetween(u,v,calc_type="COS"):
    #this calculates the angle between two vectors using either arccos, arcsing or arctan

    if calc_type=="COS": return np.arccos(dot(u,v))
    elif calc_type=="SIN": return np.arcsin(norm(cross(u,v)))
    elif calc_type=="TAN": return np.arctan2(norm(cross(u,v)), dot(u,v))

#############################################################
#                     DIMER ID FUNCTIONS                    #
#############################################################

def inter_part_distance(p1, p2):
  #particles are a list of positions (x, y, z)
  return np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def find_neighbors(particle_list, particle_id, cutoff=16):
  #this will take a particle and see which particles have a COM within some cutoff distance
  neighbor_list = []

  for o_part_id in range(particle_id, particle_list):
    distance = inter_part_distance(particle_list[particle_id], particle_list[o_part_id])

    if distance<0.0000001: continue  #this makes sure we are not checking the same particle

    if distance<cutoff:
      neighbor_list.append([particle_id, o_part_id])

  return neighbor_list

def get_specific_type_position(snap, part_id, N_p, vpp, type_id):
  #we need to recall that each particle is located in the range " N_p + part_id*vpp : N_p + part_id*vpp + vpp"
  #within this range, we need to look at a particular index associated with the type_id
  loc_type_id  = np.where(np.array(snap.particles.types)==type_id)[0][0]
  loc_particle = N_p + part_id*vpp + loc_type_id

  return snap.particles.position[loc_particle]

def create_dimer_list(snap, side_id, N_p, vpp):  #WILL NEED TO ADD A FLAG ABOUT WHICH INTERACTIONS TO CHECK
  particle_coms = snap.particles.position[:N_p]

  particle_pairs_to_check = []

  for i in range(len(particle_coms)):
    neighbor_list = find_neighbors(particle_coms, i)

    for nn_id in neighbor_list: particle_pairs_to_check.append(nn_id)

  #now we have our list of particles. At this point we need to see if they are align as a dimer
  #to do this we will need to find the positions of certain interaction sites

  temp_check_list = particle_pairs_to_check
  particle_pairs_to_check = []

  for particle_pair in temp_check_list:
    #THIS CAN BE EXTENDED TO CHECK THAT MORE OF THE INTERACTIONS ARE WITHIN A CERTAIN DISTANCE
    particle_1 = get_specific_type_position(snap, particle_pair[0], N_p, vpp, 'A'+str(side_id))   #position A1 interacts with F1 ####################################################
    particle_2 = get_specific_type_position(snap, particle_pair[1], N_p, vpp, 'F'+str(side_id))   #position F1 interacts with A1 ####################################################

    if inter_part_distance(particle_1, particle_2) < int_cutoff:
      particle_pairs_to_check.append(particle_pair)


#############################################################
#                   RELATIVE ORI FUNCTIONS                  #
#############################################################

def body_coord_sys(snap, particle_id, side_id, N_p, vpp):
  #given the coordinates we currently have we can get our y-axis from the interactions on side 1 and we
  #can get our x-axis from the COM and the COM of the side1 interactions

  #first get the coordinates of all of the interactions
  int_sideA_types = ["A1", "B1", "C1", "D1", "E1", "F1"]
  int_sideB_types = ["A2", "B2", "C2", "D2", "E2", "F2"]
  int_sideC_types = ["A3", "B3", "C3", "D3", "E3", "F3"]

  if side_id==1:
     int_side1_types, int_side2_types, int_side3_types = int_sideA_types, int_sideB_types, int_sideC_types
  if side_id==2:
     int_side1_types, int_side2_types, int_side3_types = int_sideB_types, int_sideC_types, int_sideA_types
  if side_id==3:
     int_side1_types, int_side2_types, int_side3_types = int_sideC_types, int_sideA_types, int_sideB_types

  def get_side_positions(int_side_types):
    side_positions = []

    for int_type in int_side_types:
      side_positions.append(get_specific_type_position(snap, particle_id, N_p, vpp, int_type))
    return np.array(side_positions)  

  side1_positions = get_side_positions(int_side1_types)
  side2_positions = get_side_positions(int_side2_types)
  side3_positions = get_side_positions(int_side3_types)

  #now that we have all of the side positions we can get some location data
  all_positions = np.concatenate([side1_positions, side2_positions, side3_positions], axis=0)
  com = np.average(all_positions, axis=0)

  com_side1 = np.average(side1_positions, axis=0)

  left_side1, right_side1 = np.average(side1_positions[:3], axis=0), np.average(side1_positions[3:], axis=0)

  #we can now define the axis directions
  y_hat = (left_side1 - right_side1)/norm(left_side1 - right_side1)
  x_hat = (com_side1 - com)/norm(com_side1 - com)
  z_hat = cross(x_hat, y_hat)

  return x_hat, y_hat, z_hat, com_side1

def projectVector(a,n):
    #a is the vector to project and n is the normal of the plane to project onto
    #a_P = a - (a.n)n
    a_p = a - dot(a,n)*n
    a_p = a_p/norm(a_p)
    return a_p

def interaction_translations(x,y,z, int_1, int_2):
    #x,y,z are the normal vectors for boudy 1
    #int_1 and int_2 are approximations of the interaction sites on the two bodies
    int_vector = int_1 - int_2
    dx, dy, dz = dot(int_vector, x), dot(int_vector, y), dot(int_vector, z)

    return np.array([dx, dy, dz])

def getRelativeCoordsProjections(snap, b1, b2, side_id, N_p, vpp):
    #b1, b2 are coordinates of the two bodies

    #first get the x,y,z are the coordinates of body 1
    x,y,z, int_1 = body_coord_sys(snap, b1, side_id, N_p, vpp)
    n1, n2, n3, int_2 = body_coord_sys(snap, b2, side_id, N_p, vpp)

    stretches = interaction_translations(x,y,z, int_1, int_2)

    n2_xy, n2_yz, n2_zx = projectVector(n1, z), projectVector(n2, x), projectVector(n3, y) #this gives: roll, twist, bend

    #need to also be carefull about the sign of the angles: do this by taking the cross product and seeing the sign wrt to the projection axis
    #to get the angles now see what their angle is with respect to the corresponding normal
    th_R = angleBetween(x,n2_xy,"TAN")
    th_T = angleBetween(y,n2_yz,"TAN")
    th_B = angleBetween(z,n2_zx,"TAN")

    sgn_R = np.sign(dot(cross(x, n2_xy), z))
    sgn_T = np.sign(dot(cross(y, n2_yz), x))
    sgn_B = np.sign(dot(cross(z, n2_zx), y))

    th_T = sgn_T*th_T
    th_R = sgn_R*th_R

    if th_T>0: th_T = th_T-np.pi
    else: th_T = th_T+np.pi

    if th_R>0: th_R = th_R-np.pi
    else: th_R = th_R+np.pi

    return stretches, th_R, th_T, th_B*sgn_B

def projection_analysis(snap, particle_pairs_to_check, side_id, N_p, vpp):

  relative_projections = []

  for pair in particle_pairs_to_check:
    
    body_1, body_2 = pair  #recall these are the index of the body
    stretches, th_R, th_T, th_B = getRelativeCoordsProjections(snap, body_1, body_2, side_id, N_p, vpp)
    relative_projections.append([stretches, th_R, th_T, th_B])

  return np.array(relative_projections)
#############################################################
#                     PLOTTING FUNCTIONS                    #
#############################################################

def plotCenters(snap, N_p):
  fig = plt.figure()
  axs = fig.add_subplot(111, projection='3d')
  
  for position in snap.particles.position[:N_p]:
    axs.scatter3D(position[0], position[1], position[2], c='k')
  plt.show()

def plotDimer(particle_pair, snap, N_p, vpp):
  fig = plt.figure()
  axs = fig.add_subplot(111, projection='3d')

  particle_id_1, particle_id_2 = particle_pair

  for position in snap.particles.position[N_p + particle_id_1*vpp:N_p+(particle_id_1+1)*vpp]:
    axs.scatter3D(position[0], position[1], position[2], c='r')

  for position in snap.particles.position[N_p + particle_id_2*vpp:N_p+(particle_id_2+1)*vpp]:
    axs.scatter3D(position[0], position[1], position[2], c='b')
  plt.show()


if __name__=="__main__":

  ############################################
  #preamble, setting up parameters of the simulation and setting boolean flags

  plot_part_centers = False #this plots the COM of particles
  plot_test_dimer = True  #this plots a single pair of dimers to make sure our neighbor finding code worked

  traj_file = './SimulationOutput/Side1/trajectory.gsd'

  side_id = 1  #this can be 1, 2, or 3

  N_p = 36*6  # number of triangles in the simulation
  vpp = 120   # number of particles per triangle

  int_cutoff=0.45  #this is the distance threshold for an interaction to check the dimer configuration

  gsd_file = gsd.hoomd.open(traj_file)
  
  all_relative_projections = []

  ############################################
  # NONE OF WHAT FOLLOWS HANDLES WRAPPING OF THE PERIODIC BOUNDARY CONDITIONS

  # WILL NEED TO MAKE THIS A FUNCTION TO GO THROUGH ALL OF THE FRAMES
  #create a snapshot of a single frame of the trajectory   

  for frame_id in range(len(gsd_file)):
    snap = gsd_file[frame_id]

    if plot_part_centers: plotCenters(snap, N_p)

    #the next thing to do is to see which particles are neighbors
    particle_pairs_to_check = create_dimer_list(snap, side_id, N_p, vpp)

    if plot_test_dimer: plotDimer(particle_pairs_to_check[0], snap, N_p, vpp)

    #now that we have a list of valid dimer particle pairs we want to calculate the angles of the body
    #to do this we will use the same code that we use for the cryo-em analysis, for this we will need
    #the vertex positions of the two particles
    relative_projections = projection_analysis(snap, particle_pairs_to_check, side_id, N_p, vpp)

    if frame_id==0: all_relative_projections = relative_projections
    else: all_relative_projections = np.concatenate((all_relative_projections, relative_projections))

  #we now have gone through all of the frames and have all the coords
