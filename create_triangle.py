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

def get_triangle(l1, l2, l3, h, a1, a2, a3, ne, ni_input,
                 v_f = 0.5,
                 equi = True):
    # Returns positions of particles for a triangle subunit
    # l1, l2, l3: length of each side
    # a1, a2, a3: angle of each side
    # h: height of the triangle
    # ns: number of particles per edge
    # ni: number of face particles (interacting)
    positions = []
    types = []

    v1 = (0,0,0)
    v2 = (l1,0,0)
    v3 = ((l1**2-l2**2+l3**2)/(2 * l1),np.sqrt(l3**2 - ((l1**2-l2**2+l3**2)/(2 * l1))**2),0)
    
    positions.append(v1)
    positions.append(v2)
    positions.append(v3)

    # plane of face between l1 and l1'
    # equation -sin(a1)y + cos(a1)z = 0
    
    # plane between l2 and l2':
    # 1. get vector from v2 to v3
    # 2. rotate 90 degrees around z axis
    # 3. rotate (a2 + 90) degrees around v2-v3
    v2v3 = [v3[0] - v2[0],v3[1] - v2[1],v3[2] - v2[2]]
    ar = np.pi/2
    v2v3rot = [np.cos(ar) * v2v3[0] - np.sin(ar) * v2v3[1], np.sin(ar) * v2v3[0] + np.cos(ar) * v2v3[1] , 0]
    ar2 = np.pi/2 + a2
    v2v3len = np.sqrt((v2v3[0])**2 + (v2v3[1])**2 + (v2v3[2])**2)
    u1 = [v2v3[0]/v2v3len, v2v3[1]/v2v3len, v2v3[2]/v2v3len]
    R00 = np.cos(ar2) + u1[0]**2*(1-np.cos(ar2))
    R01 = u1[0]*u1[1]*(1-np.cos(ar2)) - u1[2]*np.sin(ar2)
    R02 = u1[0]*u1[2]*(1-np.cos(ar2)) + u1[1]*np.sin(ar2)

    R10 = u1[0]*u1[1]*(1-np.cos(ar2)) + u1[2]*np.sin(ar2)
    R11 = np.cos(ar2) + u1[1]**2*(1-np.cos(ar2))
    R12 = u1[1]*u1[2]*(1-np.cos(ar2)) - u1[0]*np.sin(ar2)
    
    R20 = u1[0]*u1[2]*(1-np.cos(ar2)) - u1[1]*np.sin(ar2)
    R21 = u1[2]*u1[1]*(1-np.cos(ar2)) + u1[0]*np.sin(ar2)
    R22 = np.cos(ar2) + u1[2]**2*(1-np.cos(ar2))

    perp2 = [R00 * v2v3rot[0] + R01 * v2v3rot[1] + R02 * v2v3rot[2],
             R10 * v2v3rot[0] + R11 * v2v3rot[1] + R12 * v2v3rot[2],
             R20 * v2v3rot[0] + R21 * v2v3rot[1] + R22 * v2v3rot[2]]

    # equation for second plane
    # perp2[0]*x + perp2[1]*y + perp2[2]*z - perp2[0]*l1 = 0

    # Now same with 3rd plane
    ar = -np.pi/2
    v1v3 = [v3[0] - v1[0],v3[1] - v1[1],v3[2] - v1[2]]
    v1v3rot = [np.cos(ar) * v1v3[0] - np.sin(ar) * v1v3[1], np.sin(ar) * v1v3[0] + np.cos(ar) * v1v3[1] , 0]

    ar2 = -np.pi/2 - a3
    v1v3len = np.sqrt((v1v3[0])**2 + (v1v3[1])**2 + (v1v3[2])**2)
    u1 = [v1v3[0]/v1v3len, v1v3[1]/v1v3len, v1v3[2]/v1v3len]
    R00 = np.cos(ar2) + u1[0]**2*(1-np.cos(ar2))
    R01 = u1[0]*u1[1]*(1-np.cos(ar2)) - u1[2]*np.sin(ar2)
    R02 = u1[0]*u1[2]*(1-np.cos(ar2)) + u1[1]*np.sin(ar2)

    R10 = u1[0]*u1[1]*(1-np.cos(ar2)) + u1[2]*np.sin(ar2)
    R11 = np.cos(ar2) + u1[1]**2*(1-np.cos(ar2))
    R12 = u1[1]*u1[2]*(1-np.cos(ar2)) - u1[0]*np.sin(ar2)
    
    R20 = u1[0]*u1[2]*(1-np.cos(ar2)) - u1[1]*np.sin(ar2)
    R21 = u1[2]*u1[1]*(1-np.cos(ar2)) + u1[0]*np.sin(ar2)
    R22 = np.cos(ar2) + u1[2]**2*(1-np.cos(ar2))

    perp3 = [R00 * v1v3rot[0] + R01 * v1v3rot[1] + R02 * v1v3rot[2],
             R10 * v1v3rot[0] + R11 * v1v3rot[1] + R12 * v1v3rot[2],
             R20 * v1v3rot[0] + R21 * v1v3rot[1] + R22 * v1v3rot[2]]

    # equation for third plane
    # perp3[0]*x + perp3[1]*y + perp3[2]*z = 0

    # Three planes coming from dihedral angles
    # -sin(a1)y + cos(a1)z = 0
    # perp2[0]*x + perp2[1]*y + perp2[2]*z - perp2[0]*l1 = 0
    # perp3[0]*x + perp3[1]*y + perp3[2]*z = 0

    # Three other vertices are solutions to the three intersections
    # with z = h

    v1p = (-(perp3[1]*h/np.tan(a1) + perp3[2]*h)/perp3[0], 
           h/np.tan(a1), 
           h)
    v2p = ((perp2[0]*l1 -  perp2[1]*h/np.tan(a1) - perp2[2]*h)/perp2[0], 
           h/np.tan(a1), 
           h)
    v3p = ((l1*perp2[0]*perp3[1] - h*perp2[2]*perp3[1] + h*perp2[1]*perp3[2])/(perp2[0]*perp3[1] - perp2[1]*perp3[0]), 
           (l1*perp2[0]*perp3[0] - h*perp2[2]*perp3[0] + h*perp2[0]*perp3[2])/(perp2[1]*perp3[0] - perp2[0]*perp3[1]), 
           h)

    positions.append(v1p)
    positions.append(v2p)
    positions.append(v3p)

    for i in range(6):
        types.append('V')

    # Finally two triangle centers
    c1 = ((v1[0] + v2[0] + v3[0])/3.0, (v1[1] + v2[1] + v3[1])/3.0, 0)
    c2 = ((v1p[0] + v2p[0] + v3p[0])/3.0, (v1p[1] + v2p[1] + v3p[1])/3.0, h)

    positions.append(c1)
    positions.append(c2)
    types.append('S')
    types.append('S')

    # center of mass
    com = (0.5*(c1[0]+c2[0]), 0.5*(c1[1]+c2[1]), h/2)

    # Now edge particles, ni must be greater than 2.
    # l1 sides:
    vec = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ne-1)
    for i in range(ne - 2):
        vh = (v1[0] + (i+1)*seg*uv[0],v1[1] + (i+1)*seg*uv[1],0)
        positions.append(vh)
        types.append('S')
    vec = [v2p[0] - v1p[0], v2p[1] - v1p[1], v2p[2] - v1p[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ne-1)
    for i in range(ne - 2):
        vh = (v1p[0] + (i+1)*seg*uv[0],v1p[1] + (i+1)*seg*uv[1],h)
        positions.append(vh)
        types.append('S')
    vec = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ne-1)
    for i in range(ne - 2):
        vh = (v1[0] + (i+1)*seg*uv[0],v1[1] + (i+1)*seg*uv[1],0)
        positions.append(vh)
        types.append('S')
    vec = [v3p[0] - v1p[0], v3p[1] - v1p[1], v3p[2] - v1p[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ne-1)
    for i in range(ne - 2):
        vh = (v1p[0] + (i+1)*seg*uv[0],v1p[1] + (i+1)*seg*uv[1],h)
        positions.append(vh)
        types.append('S')
    vec = [v2[0] - v3[0], v2[1] - v3[1], v2[2] - v3[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ne-1)
    for i in range(ne - 2):
        vh = (v3[0] + (i+1)*seg*uv[0],v3[1] + (i+1)*seg*uv[1],0)
        positions.append(vh)
        types.append('S')
    vec = [v2p[0] - v3p[0], v2p[1] - v3p[1], v2p[2] - v3p[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ne-1)
    for i in range(ne - 2):
        vh = (v3p[0] + (i+1)*seg*uv[0],v3p[1] + (i+1)*seg*uv[1],h)
        positions.append(vh)
        types.append('S')

    # three particles at the middle of each side
    s1 = ((v1[0] + v1p[0])/2.0, (v1[1] + v1p[1])/2.0, h/2)
    s2 = ((v2[0] + v2p[0])/2.0, (v2[1] + v2p[1])/2.0, h/2)
    s3 = ((v3[0] + v3p[0])/2.0, (v3[1] + v3p[1])/2.0, h/2)
    positions.append(s1)
    positions.append(s2)
    positions.append(s3)
    types.append('S')
    types.append('S')
    types.append('S')

    # interaction particles
    ni = ni_input + 2
    # v_f = 0.5 # 0: 0 height. 1: h height
    # equi = True
    v_r = 1 - v_f
    v1h = [(v1[0]*v_f + v1p[0]*v_r), (v1[1]*v_f + v1p[1]*v_r), (v1[2]*v_f + v1p[2]*v_r)]
    v2h = [(v2[0]*v_f + v2p[0]*v_r), (v2[1]*v_f + v2p[1]*v_r), (v2[2]*v_f + v2p[2]*v_r)]
    v3h = [(v3[0]*v_f + v3p[0]*v_r), (v3[1]*v_f + v3p[1]*v_r), (v3[2]*v_f + v3p[2]*v_r)]
    
    vec = [v2h[0] - v1h[0], v2h[1] - v1h[1], v2h[2] - v1h[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ni-1)
    if equi == True:
        seg = length / ni_input
        for i in range(ni-2):
            vh = (v1h[0] + seg/2 * uv[0] + i*seg*uv[0],v1h[1] + seg/2 * uv[1] + i*seg*uv[1] , v1h[2])
            positions.append(vh)
    else:
        for i in range(ni - 2):
            vh = (v1h[0] + (i+1)*seg*uv[0],v1h[1] + (i+1)*seg*uv[1],v1h[2])
            positions.append(vh)

    vec = [v2h[0] - v3h[0], v2h[1] - v3h[1], v2h[2] - v3h[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ni-1)
    if equi == True:
        seg = length / ni_input
        for i in range(ni-2):
            vh = (v3h[0] + seg/2 * uv[0] + i*seg*uv[0],v3h[1] + seg/2 * uv[1] + i*seg*uv[1] , v3h[2])
            positions.append(vh)
    else:
        for i in range(ni - 2):
            vh = (v3h[0] + (i+1)*seg*uv[0],v3h[1] + (i+1)*seg*uv[1],v3h[2])
            positions.append(vh)

    vec = [v3h[0] - v1h[0], v3h[1] - v1h[1], v3h[2] - v1h[2]]
    length = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    uv = vec/length
    seg = length/(ni-1)
    if equi == True:
        seg = length / ni_input
        for i in range(ni-2):
            vh = (v1h[0] + seg/2 * uv[0] + i*seg*uv[0],v1h[1] + seg/2 * uv[1] + i*seg*uv[1] , v1h[2])
            positions.append(vh)
    else:
        for i in range(ni - 2):
            vh = (v1h[0] + (i+1)*seg*uv[0],v1h[1] + (i+1)*seg*uv[1],v1h[2])
            positions.append(vh)

    # center everything to (0,0,0)
    for i in range(len(positions)):
        positions[i] = (positions[i][0] - com[0], positions[i][1]-com[1], positions[i][2]-com[2])
    
    return [positions,types]

def plotTriangle(xcoords, ycoords, zcoords, types):
    fig = plt.figure()
    ax = a3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    for i in range(0, len(xcoords)):
        if types[i] == 'V':
            ax.scatter(xcoords[i], ycoords[i], zcoords[i], color='blue')
        elif types[i] == 'S':
            ax.scatter(xcoords[i], ycoords[i], zcoords[i], color='green')
        else:
            ax.scatter(xcoords[i], ycoords[i], zcoords[i], color='red')
        ax.text(xcoords[i], ycoords[i], zcoords[i], types[i], None)
            
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    f1 = [[
        [xcoords[0],ycoords[0],zcoords[0]],
                   [xcoords[3],ycoords[3],zcoords[3]],
                   [xcoords[4],ycoords[4],zcoords[4]],
                   [xcoords[1],ycoords[1],zcoords[1]]
        ]]
    
    f2 = [[
        [xcoords[1],ycoords[1],zcoords[1]],
                   [xcoords[4],ycoords[4],zcoords[4]],
                   [xcoords[5],ycoords[5],zcoords[5]],
                   [xcoords[2],ycoords[2],zcoords[2]]
    ]
         ]
    
    f3 = [[
                   [xcoords[0],ycoords[0],zcoords[0]],
                   [xcoords[2],ycoords[2],zcoords[2]],
                   [xcoords[5],ycoords[5],zcoords[5]],
                   [xcoords[3],ycoords[3],zcoords[3]]
    ]
         ]
    
    f4 = [[
        [xcoords[3],ycoords[3],zcoords[3]],
                   [xcoords[5],ycoords[5],zcoords[5]],
                   [xcoords[4],ycoords[4],zcoords[4]]
    ]
         ]
    
    f5 = [[
        [xcoords[0],ycoords[0],zcoords[0]],
                   [xcoords[1],ycoords[1],zcoords[1]],
                   [xcoords[2],ycoords[2],zcoords[2]]
    ]
         ]
    ax.add_collection3d(Poly3DCollection(f1, edgecolor='black', linewidths=1, alpha=0.2))
    ax.add_collection3d(Poly3DCollection(f2, edgecolor='black', linewidths=1, alpha=0.2))
    ax.add_collection3d(Poly3DCollection(f3, edgecolor='black', linewidths=1, alpha=0.2))
    ax.add_collection3d(Poly3DCollection(f4, edgecolor='black', linewidths=1, alpha=0.2))
    ax.add_collection3d(Poly3DCollection(f5, edgecolor='black', linewidths=1, alpha=0.2))
    ax.view_init(azim=45, elev=30)

def fixPeriodic(V, N, L):
    #vertices are potentially on opposite sides of domain b/c of periodicity. fix this
    #max distance in pentagon is phi=1.618
    #attractors are distance 1.175 apart, so max dist is 1.9
    max_diff = 3.0  #max distance allowed for particles in the rigid pentagon 
    #loop over each body, compare coordinates of each vertex to the first, check if bigger
    #than the max allowed
    for body in range(N):
        for vertex in range(1,6):
            for d in range(3):
                diff = V[body][vertex][d] - V[body][0][d]
                if (np.abs(diff) > max_diff):
                    s = diff / np.abs(diff)
                    V[body][vertex][d] += -s*L[d]
    return
    
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return

def plotFrame_3d_full(file, frame, snapshot=None, topng=False, shift = (0,0,0)):
    if snapshot is not None:
        snap = snapshot
    else:
        gsd_file = gsd.hoomd.open(file)
        snap = gsd_file[frame]
    #get the box data
    box = snap.configuration.box
    L = [box[0], box[1], box[2]]
    #get total number of particles
    Ntot = snap.particles.N
    #get index of 'V' type particles
    A_index = np.where(np.array(snap.particles.types) == 'V')[0][0]
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
    fixPeriodic(V,N,L)
    M = np.amax(np.amax(V,axis=1),axis=0)
    m = np.amin(np.amin(V,axis=1),axis=0)
    #loop over the bodies and construct pentagons using the vertices, fill, plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for body in range(N):
        hull = ConvexHull(np.array(V[body]))
        for s in hull.simplices:
            tri = Poly3DCollection([V[body][s]])
            tri.set_color('green')
            tri.set_edgecolor('blue')
            tri.set_alpha(0.5)
            ax.add_collection3d(tri)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_zlim(-15, 15)
    ax.view_init(35, -80)
    set_axes_equal(ax)
    if topng == False:
        plt.show()
    else:
        plt.savefig('pngs/img_'+str(frame).zfill(7)+'.png')
        plt.close(fig)
    return


def plotFrame_3d(file, frame, snapshot=None, topng=False, shift = (0,0,0)):
    if snapshot is not None:
        snap = snapshot
    else:
        gsd_file = gsd.hoomd.open(file)
        snap = gsd_file[frame]
    #get the box data
    box = snap.configuration.box
    L = [box[0], box[1], box[2]]
    #get total number of particles
    Ntot = snap.particles.N
    #get index of 'V' type particles
    A_index = np.where(np.array(snap.particles.types) == 'V')[0][0]
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
    fixPeriodic(V,N,L)
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
    set_axes_equal(ax)
    if topng == False:
        plt.show()
    else:
        plt.savefig('pngs/img_'+str(frame).zfill(7)+'.png')
        plt.close(fig)
    return




















