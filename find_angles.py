import numpy as np
import sys

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

def inter_part_distance(p1, p2, box_size):
  #particles are a list of positions (x, y, z)

  #need to account for the periodic box
  dx, dy, dz = p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]
  dx = np.min([dx, box_size-dx])
  dy = np.min([dy, box_size-dy])
  dz = np.min([dz, box_size-dz])
    
  return np.sqrt( dx**2 + dy**2 + dz**2)

def find_neighbors(particle_list, particle_id, box_size, cutoff=4):
  #this will take a particle and see which particles have a COM within some cutoff distance
  neighbor_list = []

  for o_part_id in range(len(particle_list)):
    distance = inter_part_distance(particle_list[particle_id], particle_list[o_part_id], box_size)

    if distance<0.0000001: continue  #this makes sure we are not checking the same particle

    if distance<cutoff:
      neighbor_list.append([particle_id, o_part_id])

  return neighbor_list

def get_specific_type_position(snap, part_id, N_p, vpp, type_id):
  #we need to recall that each particle is located in the range " N_p + part_id*vpp : N_p + part_id*vpp + vpp"
  #within this range, we need to look at a particular index associated with the type
  idx_left = N_p + part_id*vpp
  idx_right = N_p + part_id*vpp + vpp

  type_index = np.where(np.array(snap.particles.types)==type_id)[0][0]
  
  #print(snap.particles.typeid[idx_left:idx_right])
  #print(np.where(np.array(snap.particles.typeid[idx_left:idx_right])==type_index)[0][0])
  
  loc_type_id  = np.where(np.array(snap.particles.typeid[idx_left:idx_right])==type_index)[0][0]
  #print(loc_type_id)
  
  loc_particle = N_p + part_id*vpp + loc_type_id

  return snap.particles.position[loc_particle]

def create_dimer_list(snap, side_id, N_p, vpp, box_size):  #WILL NEED TO ADD A FLAG ABOUT WHICH INTERACTIONS TO CHECK
  particle_coms = snap.particles.position[:N_p]

  particle_pairs_to_check = []

  print('creating dimer list')
  
  for i in range(len(particle_coms)):
    neighbor_list = find_neighbors(particle_coms, i, box_size)
    
    for nn_id in neighbor_list: particle_pairs_to_check.append(nn_id)

  #now we have our list of particles. At this point we need to see if they are align as a dimer
  #to do this we will need to find the positions of certain interaction sites

  #print(particle_pairs_to_check)
  
  temp_check_list = particle_pairs_to_check
  particle_pairs_to_check = []

  for particle_pair in temp_check_list:
    #THIS CAN BE EXTENDED TO CHECK THAT MORE OF THE INTERACTIONS ARE WITHIN A CERTAIN DISTANCE
    particle_1_A = get_specific_type_position(snap, particle_pair[0], N_p, vpp, 'A'+str(side_id))   #position A1 interacts with F1 ####################################################
    particle_2_F = get_specific_type_position(snap, particle_pair[1], N_p, vpp, 'F'+str(side_id))   #position F1 interacts with A1 ####################################################

    particle_1_F = get_specific_type_position(snap, particle_pair[0], N_p, vpp, 'F'+str(side_id))   #position A1 interacts with F1 ####################################################
    particle_2_A = get_specific_type_position(snap, particle_pair[1], N_p, vpp, 'A'+str(side_id))   #position F1 interacts with A1 ####################################################

    #print(particle_pair)
    #print(particle_1, particle_2)
    
    if inter_part_distance(particle_1_A, particle_2_F, box_size) < int_cutoff:
      if inter_part_distance(particle_1_F, particle_2_A, box_size) < int_cutoff:
      
        particle_pairs_to_check.append(particle_pair)

  #print(particle_pairs_to_check)

  return particle_pairs_to_check

#############################################################
#                   RELATIVE ORI FUNCTIONS                  #
#############################################################

def body_coord_sys(snap, particle_id, side_id, N_p, vpp, box_size, plotParticle=False):
  #given the coordinates we currently have we can get our y-axis from the interactions on side 1 and we
  #can get our x-axis from the COM and the COM of the side1 interactions

  box_shift = np.array([-box_size/2., -box_size/2., -box_size/2.])
    
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

  p_com = np.asarray(snap.particles.position[particle_id])
     
  def get_side_positions(int_side_types):
    side_positions = []

    for int_type in int_side_types:
      position_to_append = get_specific_type_position(snap, particle_id, N_p, vpp, int_type)
      position_to_append = position_to_append - box_shift-p_com
      side_positions.append(position_to_append)
    return np.array(side_positions)  

  side1_positions = (get_side_positions(int_side1_types))%(box_size) + box_shift + p_com
  side2_positions = (get_side_positions(int_side2_types))%(box_size) + box_shift + p_com
  side3_positions = (get_side_positions(int_side3_types))%(box_size) + box_shift + p_com

  #now that we have all of the side positions we can get some location data
  all_positions = np.concatenate([side1_positions, side2_positions, side3_positions], axis=0)
  com = np.average(all_positions, axis=0)

  com_side1 = np.average(side1_positions, axis=0)

  if plotParticle:

      #print(np.min(side1_positions), np.max(side1_positions))
      #print(np.min(side2_positions), np.max(side2_positions))
      #print(np.min(side3_positions), np.max(side3_positions))
      
      fig = plt.figure()
      axs = fig.add_subplot(111, projection='3d')

      axs.scatter3D(all_positions.T[0], all_positions.T[1], all_positions.T[2], c='k', alpha=0.4)
      axs.scatter3D(com[0], com[1], com[2], c='r')

      axs.set_aspect('equal')
      
      plt.show()
      plt.close()
  
  left_side1, right_side1 = np.average(side1_positions[:3], axis=0), np.average(side1_positions[3:], axis=0)

  #we can now define the axis directions
  y_hat = -(left_side1 - right_side1)/norm(left_side1 - right_side1)
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
    int_vector = int_2 - int_1
    dx, dy, dz = dot(int_vector, x), dot(int_vector, y), dot(int_vector, z)

    return np.array([dx, dy, dz])

def getRelativeCoordsProjections(snap, b1, b2, side_id, N_p, vpp, box_size):
    #b1, b2 are coordinates of the two bodies

    #first get the x,y,z are the coordinates of body 1
    x,y,z, int_1 = body_coord_sys(snap, b1, side_id, N_p, vpp, box_size)
    n1, n2, n3, int_2 = body_coord_sys(snap, b2, side_id, N_p, vpp, box_size)

    stretches = interaction_translations(x,y,z, int_1, int_2)

    n2_xy, n2_yz, n2_zx = projectVector(n1, z), projectVector(n2, x), projectVector(n3, y) #this gives: roll, twist, bend

    '''
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
    th_B = sgn_B*th_B

    if th_T>0: th_T = th_T-np.pi
    else: th_T = th_T+np.pi

    if th_R>0: th_R = th_R-np.pi
    else: th_R = th_R+np.pi
    '''

    th_R = angleBetween(y, n2_xy, "TAN") - np.pi/2.
    th_T = angleBetween(z, n2_yz, "TAN") - np.pi/2. 
    th_B = angleBetween(x, n2_zx, "TAN") - np.pi/2.
    
    return stretches, th_R, th_T, th_B

def projection_analysis(snap, particle_pairs_to_check, side_id, N_p, vpp, box_size):

  relative_projections = []

  for pair in particle_pairs_to_check:

    #print(pair)
      
    body_1, body_2 = pair  #recall these are the index of the body
    stretches, th_R, th_T, th_B = getRelativeCoordsProjections(snap, body_1, body_2, side_id, N_p, vpp, box_size)
    dx, dy, dz = stretches
    relative_projections.append([dx, dy, dz, th_R, th_T, th_B])

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

def plotDimer(particle_pair, snap, side_id, N_p, vpp, box_size):
    
  #this vector will shift all of the particle positions to be positive
  box_shift = np.array([-box_size/2., -box_size/2., -box_size/2.])

  particle_id_1, particle_id_2 = particle_pair

  particle_1_com = snap.particles.position[particle_id_1] #- box_shift
  particle_2_com = snap.particles.position[particle_id_2] #- box_shift
  
  #move all the points by a single displacement vector
  displacement_vec = np.average([particle_1_com, particle_2_com], axis=0)

  #print(displacement_vec)
  #print(snap.particles.position[particle_id_1])
  
  avg_com = np.average(np.array([particle_1_com, particle_2_com]), axis=0)

  #get the coord vectors

  x1, y1, z1, com1 = body_coord_sys(snap, particle_id_1, side_id, N_p, vpp, box_size, True)

  x2, y2, z2, com2 = body_coord_sys(snap, particle_id_2, side_id, N_p, vpp, box_size, True)

  #print(x1, y1, z1)
  #print(x2, y2, z2)

  print("check com")
  print(particle_1_com)

  
  part1_range_L, part1_range_R = N_p + particle_id_1*vpp, N_p+(particle_id_1+1)*vpp
  part2_range_L, part2_range_R = N_p + particle_id_2*vpp, N_p+(particle_id_2+1)*vpp

  #for the different particles we need to do a check to make sure that the points are in not
  # being broken up by the periodic boundaries

  part_1_v_pos  = np.asarray(snap.particles.position[part1_range_L:part1_range_R-18]) - box_shift
  part_1_int_pos= np.asarray(snap.particles.position[part1_range_R-18:part1_range_R]) - box_shift

  part_2_v_pos  = np.asarray(snap.particles.position[part2_range_L:part2_range_R-18]) - box_shift
  part_2_int_pos= np.asarray(snap.particles.position[part2_range_R-18:part2_range_R]) - box_shift

  part_1_v_pos   = (part_1_v_pos - displacement_vec)%box_size + box_shift + displacement_vec
  part_1_int_pos = (part_1_int_pos - displacement_vec)%box_size + box_shift + displacement_vec
  part_2_v_pos   = (part_2_v_pos - displacement_vec)%box_size + box_shift + displacement_vec
  part_2_int_pos = (part_2_int_pos - displacement_vec)%box_size + box_shift + displacement_vec

  #print(part_1_v_pos)
  
  def plotPoints(position, c='r'):
    axs.scatter3D(position.T[0], position.T[1], position.T[2], c=c)

  fig = plt.figure()
  axs = fig.add_subplot(111, projection='3d')

  plotPoints(part_1_v_pos, c='r')
  plotPoints(part_1_int_pos, c='orange')
  plotPoints(part_2_v_pos, c='b')
  plotPoints(part_2_int_pos, c='cyan')
  
  for normal_vec in [x1, y1, z1]:
     #origin  = particle_1_com - avg_com + displacement_vec
     #vec_end = particle_1_com - avg_com + normal_vec + displacement_vec
     origin  = com1 - avg_com + displacement_vec
     vec_end = com1 - avg_com + normal_vec + displacement_vec
     axs.plot([origin[0], vec_end[0]], [origin[1], vec_end[1]], [origin[2], vec_end[2]], c='r')

     
  for normal_vec in [x2, y2, z2]:
     #origin  = particle_2_com - avg_com + displacement_vec
     #vec_end = particle_2_com - avg_com + normal_vec + displacement_vec
     origin  = com2 - avg_com + displacement_vec
     vec_end = com2 - avg_com + normal_vec + displacement_vec
     axs.plot([origin[0], vec_end[0]], [origin[1], vec_end[1]], [origin[2], vec_end[2]], c='b')


  axs.set_aspect('equal')
  
  plt.show()
  plt.close()


if __name__=="__main__":
    
  ############################################
  #preamble, setting up parameters of the simulation and setting boolean flags

  plot_part_centers = False #this plots the COM of particles
  plot_test_dimer = False #this plots a single pair of dimers to make sure our neighbor finding code worked

  if len(sys.argv)>1:
    side_id = int(sys.argv[1])  #this can be 1, 2, or 3
    src = sys.argv[2]
  else: side_id=1
    
  #traj_file = './SimulationOutput/Side'+str(side_id)+'/trajectory.gsd'
  #output_src = './SimulationOutput/Side'+str(side_id)+'/'

  traj_file = src+'trajectory.gsd'
  output_src = src

  N_p = 72  # number of triangles in the simulation
  vpp = 92   # number of particles per triangle
  box_size=22 #size of simulation box
  
  int_cutoff=0.7  #this is the distance threshold for an interaction to check the dimer configuration

  gsd_file = gsd.hoomd.open(traj_file)
  
  all_relative_projections = []

  ############################################
  # NONE OF WHAT FOLLOWS HANDLES WRAPPING OF THE PERIODIC BOUNDARY CONDITIONS

  # WILL NEED TO MAKE THIS A FUNCTION TO GO THROUGH ALL OF THE FRAMES
  #create a snapshot of a single frame of the trajectory   

  for frame_id in range(len(gsd_file))[::-1]:
  
    print('analyzing frame ', frame_id,'/',len(gsd_file))
    
    snap = gsd_file[frame_id]
    
    if plot_part_centers: plotCenters(snap, N_p)

    #the next thing to do is to see which particles are neighbors
    particle_pairs_to_check = create_dimer_list(snap, side_id, N_p, vpp, box_size)
    
    print(len(particle_pairs_to_check))

    if plot_test_dimer: plotDimer(particle_pairs_to_check[0], snap, side_id, N_p, vpp, box_size)

    #now that we have a list of valid dimer particle pairs we want to calculate the angles of the body
    #to do this we will use the same code that we use for the cryo-em analysis, for this we will need
    #the vertex positions of the two particles
    relative_projections = projection_analysis(snap, particle_pairs_to_check, side_id, N_p, vpp, box_size)

    for rp in relative_projections: all_relative_projections.append(rp)
    
  #we now have gone through all of the frames and have all the coords
  all_relative_projections = np.array(all_relative_projections)

  #print(all_relative_projections.T[-1])

  np.save(output_src+'side'+str(side_id)+'_fluctuations.npy', all_relative_projections)
