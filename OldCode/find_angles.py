import numpy as np

import matplotlib
# matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

import hoomd, gsd.hoomd
import itertools
import math
import datetime
import os

import warnings
import matplotlib.colors as colors

def find_angles_general(file, frame, snapshot=None, topng=False, shift = (0,0,0), side_id=1):
    
    N_p = 36*6 #number of triangles in the simulation, specified from the params.py file
    
    #vpp = 39   #this is because in the original model particle there were 39 paritlces per triangel (30 V + 9 int)
    vpp = 120   #this is for the "low density" triangle model

    if snapshot is not None:
        snap = snapshot
    else:
        gsd_file = gsd.hoomd.open(file)
        snap = gsd_file[frame]
    
    #get the box data
    box = snap.configuration.box
    L = [box[0], box[1], box[2]]  #this is the size of the simulation box

    #get total number of particles
    Ntot = snap.particles.N
    
    #interaction names of partilces

    #get index of 'V' type particles   THIS I WILL NEED TO CHANGE TO ACCOUNT FOR THE NEW INTERACTIONS
    A_index = np.where(np.array(snap.particles.types) == 'A'+str(int(side_id)))[0][0]
    B_index = np.where(np.array(snap.particles.types) == 'B'+str(int(side_id)))[0][0]
    C_index = np.where(np.array(snap.particles.types) == 'C'+str(int(side_id)))[0][0]
    D_index = np.where(np.array(snap.particles.types) == 'D'+str(int(side_id)))[0][0]
    E_index = np.where(np.array(snap.particles.types) == 'E'+str(int(side_id)))[0][0]
    F_index = np.where(np.array(snap.particles.types) == 'F'+str(int(side_id)))[0][0]

    
    pair_n = 0
    pairs = np.array([[]])
    angles = []
    done = False
    all_types = snap.particles.types
    indexA = 0
    indexC = 0

    for i in range(len(all_types)):
        if all_types[i] == 'A':
            indexC = i
        if all_types[i] == 'F':
            indexA = i

    for i in range(N_p):
        if done:
            break;
        f_i = N_p + i*vpp
        l_i = N_p + i*vpp + vpp

        com1  = np.array([0.0,0.0,0.0])
        com1 += snap.particles.position[i]

        for j in range(N_p - i - 1):
            if done:
                break;
            f_ij = N_p + (i+j+1)*vpp
            l_ij = N_p + (i+j+1)*vpp + vpp

            com2  = np.array([0.0,0.0,0.0])
            com2 += snap.particles.position[i + j + 1]
            dist = com2 - com1
            distance_sq = dist[0]**2 + dist[1]**2 + dist[2]**2

            if distance_sq < 16:
                posA1 = np.array([0.0,0.0,0.0])
                posA2 = np.array([0.0,0.0,0.0])
                ind1 = 0
                ind2 = 0

                for k1 in range(vpp):
                    if snap.particles.typeid[f_ij + k1] == indexC:
                        ind1 = k1

                for k1 in range(vpp):
                    if snap.particles.typeid[f_i + k1] == indexA:
                        ind2 = k1

                posA1 += snap.particles.position[f_i + ind2]
                posA2 += snap.particles.position[f_ij + ind1]
                distAs = posA1 - posA2
                distAf = np.sqrt(distAs[0]**2 + distAs[1]**2 + distAs[2]**2)

                if distAf < 0.45:
                    pair_n += 1
                    # This represent a dimer
                    # print('Dimer')
                    # print(snap.particles.position[f_i])
                    # print(snap.particles.position[f_ij])
                    p1 = snap.particles.position[f_i] - snap.particles.position[f_ij+1]
                    # print(np.sqrt(p1[0]**2 + p1[1]**2 + p1[2]**2))
                    # print(snap.particles.position[f_i+1])
                    # print(snap.particles.position[f_ij+1])
                    p1 = snap.particles.position[f_i+1] - snap.particles.position[f_ij]
                    # print(np.sqrt(p1[0]**2 + p1[1]**2 + p1[2]**2))
                    # print(snap.particles.position[f_i+2])
                    # print(snap.particles.position[f_ij+2])
                    p1 = snap.particles.position[f_i+2] - snap.particles.position[f_ij+2]
                    # print(np.sqrt(p1[0]**2 + p1[1]**2 + p1[2]**2))

                    v = snap.particles.position[f_i] - snap.particles.position[f_i + 5]
                    w = snap.particles.position[f_i] - snap.particles.position[f_i + 10]
                    nx = v[1]*w[2] - v[2]*w[1]
                    ny = v[2]*w[0] - v[0]*w[2]
                    nz = v[0]*w[1] - v[1]*w[0]

                    v = snap.particles.position[f_ij] - snap.particles.position[f_ij + 5]
                    w = snap.particles.position[f_ij] - snap.particles.position[f_ij + 10]
                    mx = v[1]*w[2] - v[2]*w[1]
                    my = v[2]*w[0] - v[0]*w[2]
                    mz = v[0]*w[1] - v[1]*w[0]

                    n = np.sqrt(nx**2 + ny**2 + nz**2)
                    m = np.sqrt(mx**2 + my**2 + mz**2)

                    alpha = 180.0 * np.arccos((nx*mx + ny*my + nz*mz)/(n*m)) / np.pi
                    # print('Angle: ' + str(alpha))
                    if alpha < 90:
                    # if alpha > 180.0:
                    #     alpha -= 360.0
                        angles = np.append(angles, alpha)

                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='3d')
                    # 
                    # pos1 = snap.particles.position[f_i:f_i+6]
                    # pos2 = snap.particles.position[f_ij:f_ij+6]
                    # xs = []
                    # ys = []
                    # zs = []
                    # xz = []
                    # yz = []
                    # zz = []
                    # for i in range(6):
                    #     xs = np.append(xs, pos1[i,0])
                    #     xz = np.append(xz, pos2[i,0])
                    #     ys = np.append(ys, pos1[i,1])
                    #     yz = np.append(yz, pos2[i,1])
                    #     zs = np.append(zs, pos1[i,2])
                    #     zz = np.append(zz, pos2[i,2])
                    # ax.scatter(xs,ys,zs, color='black')
                    # ax.scatter(xz,yz,zz, color='red')
                    # print(alpha)
                    # # ax.set_xlim([])
                    # plt.show()
                    #
                    # done = True

    # print('Pairs: ' + str(pair_n) + ' found')
    # print('Pairs of A:')
    # print(pairs)
    #
    # print(angles)
    # print(np.mean(angles))
    # print(np.std(angles))
    return angles

    #get the number of particles of type A
    particle_types = snap.particles.typeid
    N = len(snap.particles.typeid[particle_types == A_index])
    #init list to store vertices
    V = np.zeros((N, 6, 3))
    #get the vertices of each pentagon via the attractor coordinates
    count = 0
    body = 0
    for particle in range(Ntot):
        if (particle_types[particle] == A_index):
            vertex = np.array(snap.particles.position[particle]) + np.array(shift)
            if vertex[0] > 15: vertex[0] -= 30
            if vertex[0] < -15: vertex[0] += 30
            if vertex[1] > 15: vertex[1] -= 30
            if vertex[1] < -15: vertex[1] += 30
            if vertex[2] > 15: vertex[2] -= 30
            if vertex[2] < -15: vertex[2] += 30
            V[body][count] = np.array(vertex)
            count += 1
        if (count == 6):
            body += 1
            count = 0
    M = np.amax(np.amax(V,axis=1),axis=0)
    m = np.amin(np.amin(V,axis=1),axis=0)
    #loop over the bodies and construct pentagons using the vertices, fill, plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for body in range(N):
        vtx1 = V[body][0:3]
        vtx2 = V[body][3:6]
        s1 = np.r_[0:2,3:5]
        s2 = np.r_[1:3,4:6]
        s3 = np.r_[0,2:4,5]
        vtx3 = V[body][s1]
        vtx4 = V[body][s2]
        vtx5 = V[body][s3]
        pent = a3.art3d.Poly3DCollection([vtx1])
        pent.set_color("g")
        pent.set_edgecolor("k")
        ax.add_collection(pent)
    #plot result
    # ax.set_xlim(m[0], M[0])
    # ax.set_ylim(m[1], M[1])
    # ax.set_zlim(m[2], M[2])
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.view_init(35, -80)
    if topng == False:
        plt.show()
    else:
        plt.savefig('pngs/img_'+str(frame).zfill(7)+'.png')
        plt.close(fig)
    return




















