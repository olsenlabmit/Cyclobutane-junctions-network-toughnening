#!/use/local/bin/env python
# -*- coding: utf-8 -*-

# Code for Generation of Network
# Follows algorithm published by 
# AA Gusev, Macromolecules, 2019, 52, 9, 3244-3251

import math
import numpy as np
from numpy import linalg as LA
import random
import ioLAMMPS

#random.seed(a=786564)


def sortSecond(val): 
    return val[1]

def bondlengths(n_chains, chains, links, Lx, Ly, Lz):
 
    dist = np.zeros((n_chains,4))
    dist[:,0:3] = chains
    dist[:,3] = -1
        
    for i in range (0, n_chains):
        if(chains[i,2] !=-1):
      
          link_1 = chains[i,1]-1
          link_2 = chains[i,2]-1
          lk = links[link_1,:] - links[link_2,:]
          
          lk[0] = lk[0] - int(round(lk[0]/Lx))*Lx
          lk[1] = lk[1] - int(round(lk[1]/Ly))*Ly
          lk[2] = lk[2] - int(round(lk[2]/Lz))*Lz
                
          dist[i,3] = LA.norm(lk)

    return dist


def find_neighbours(link_1, n_links, links, conn, cnt_length, mean_length, Lx, Ly, Lz):

    neigh = []
    lk    = np.zeros((n_links,3), dtype = float)
    
    lk[:,0] = links[:,0] - links[link_1,0]    
    lk[:,1] = links[:,1] - links[link_1,1]    
    lk[:,2] = links[:,2] - links[link_1,2]    

    for i in range (0, n_links):
       
        lk[i,0] = lk[i,0] - int(round(lk[i,0]/Lx))*Lx
        lk[i,1] = lk[i,1] - int(round(lk[i,1]/Ly))*Ly
        lk[i,2] = lk[i,2] - int(round(lk[i,2]/Lz))*Lz
                
         
    dist = LA.norm(lk, axis=1)
    dist_list = np.where(dist[:] <= cnt_length)[0]
 
    # Checking what all linkers are empty 
    for i in range(0, np.size(dist_list)):
        lnk = dist_list[i]
        a = np.where(conn[lnk,:] == -1)[0]
        if(np.size(a) > 0):
           p = math.exp(-1.5*dist[lnk]**2/mean_length)
           neigh.append([lnk, dist[lnk], p])

    neigh.sort(key = sortSecond) 

    return neigh


def find_second_link(neigh):
    
    nbr  = np.zeros(len(neigh), dtype = int)
    prob = np.zeros(len(neigh), dtype = float)
    cum_prob = np.zeros(len(neigh)+1, dtype = float)

    for i in range(0, len(neigh)):
        nbr[i]  = neigh[i][0]
        prob[i] = neigh[i][2]

 
    prob = prob/np.sum(prob)
    cum_prob[0] = 0
    for i in range(0, len(prob)):
        cum_prob[i+1] = cum_prob[i] + prob[i]

    rnd_num = random.uniform(0,1)
    a1 = np.where(cum_prob <= rnd_num)[0]

    second_link = neigh[a1[len(a1)-1]][0]

    return second_link



def generate_network(prob, func, N, L, l0, n_chains, n_links):

    print('----------------------------')
    print('-----Generating Network-----')
    print('----------------------------')
        
    cnt_length = N*l0
    mean_length = N*l0*l0
    Lx = L; Ly = L; Lz = L
    
    # Array sizes and initializations
    links   = np.zeros((n_links,3), dtype = float)
    chains  = np.full((n_chains,3), -1, dtype = int)
    conn    = np.full((n_links,func), -1, dtype = int)
    
    
    for i in range (0, n_links):
        x = random.uniform(0, Lx)
        y = random.uniform(0, Ly)
        z = random.uniform(0, Lz)
        links[i,:] = [x, y, z]
    
    for i in range (0, n_chains):
    
        chains[i,0] = N   
        found = 0
        while (found == 0): 
              link_1 = random.randint(0, n_links-1)
              a1 = np.where(conn[link_1,:] == -1)[0]
              if(np.size(a1)>0):
                 found = 1
    
        rnd_num = random.uniform(0,1)
        if (prob > rnd_num):
           chains[i,1] = link_1
           a1 = np.where(conn[link_1,:] == -1)[0]
           conn[link_1, a1[0]] = n_links + 1    
           # This ensure that link occ is filled
           # temporary filing by n_links + 1
           # later, it will be replaced by link 2 if it needs to be connected. 
           # Otherwise, it will be as it is, indicating that this site 
           # contains a dangling end.
    
           rnd_num = random.uniform(0,1)
           neigh = find_neighbours(link_1, n_links, links, conn, cnt_length, mean_length, Lx, Ly, Lz)
           if (prob > rnd_num and len(neigh) > 0):
              link_2 = find_second_link(neigh)
    
              chains[i,2] = link_2
              a2 = np.where(conn[link_2,:] == -1)[0]
              conn[link_1, a1[0]] = link_2
              conn[link_2, a2[0]] = link_1
    
    
    a1 = np.where(chains[:,1]==-1)[0]
    a2 = np.where(chains[:,2]==-1)[0]
    n_free = np.size(a1)
    n_dang = np.size(a2)-np.size(a1)
    n_connected = n_chains - np.size(a2)
    
    n_loops = 0
    all_loop_atoms = []
    for i in range (0, n_chains):
        if(chains[i,2] !=-1):
           if(chains[i,1] == chains[i,2]):
             n_loops = n_loops + 1
             all_loop_atoms.append(chains[i,1])

    all_loop_atoms.sort()
    all_loop_atoms.append(-1)
 
    i = 0
    loop_atoms = []
    dumbbell_atoms = []
    while (i < len(all_loop_atoms)-1):
        if (all_loop_atoms[i] == all_loop_atoms[i+1]):
           dumbbell_atoms.append(all_loop_atoms[i])
           i = i + 2
        else:
           loop_atoms.append(all_loop_atoms[i])
           i = i + 1
           
           
    f1 = open('all_loops','w')
    for i in range(0, len(all_loop_atoms)):
        f1.write('%5i  %5i\n'%(i, all_loop_atoms[i])) 
    f1.close()

    f1 = open('primary_loops','w')
    for i in range(0, len(loop_atoms)):
        f1.write('%5i  %5i\n'%(i, loop_atoms[i])) 
    f1.close()

    f1 = open('dumbbells','w')
    for i in range(0, len(dumbbell_atoms)):
        f1.write('%5i  %5i\n'%(i, dumbbell_atoms[i])) 
    f1.close()

 
    # new_chain are the chains excluding all the loops, dangling, and sol (free) 
    new_chains = np.zeros((n_chains - n_loops - n_dang - n_free, 3), dtype = int)
    cnt = 0 
    for i in range (0, n_chains):
        if(chains[i,2]!=-1):
           if(chains[i,1]!=chains[i,2]):
             new_chains[cnt,:] = chains[i,:]
             cnt = cnt + 1

    # Upto now, chains (or new chains) contain link with indices from 0 to 1199
    # But, for writing LAMMPS file as well as for calculating distances
    # chains are modified to have linkers with indices starting from 1 to 1200
    new_chains[:,1] = new_chains[:,1] + 1    
    new_chains[:,2] = new_chains[:,2] + 1    

    
    print('----------------------------')
    print('--Network Characteristics---')
    print('----------------------------')
    file1=open("net_char","w")
    file1.write("Box Length = %6.4f\n" %(L))
    file1.write("Contour Length = %i\n" %(N))
    file1.write("No. of strands = %i\n" %(n_chains))
    file1.write("No. of linkers = %i\n" %(n_links))
    file1.write("Extent of reaction = %6.2f\n" %(prob))
    file1.write("Free chains = %i\n" %(n_free))
    file1.write("Dangling ends = %i\n" %(n_dang))
    file1.write("Connected chains = %i\n" %(n_connected))
    file1.write('No. of primary loops = %i\n' %(n_loops))
    file1.close()


##    dist = bondlengths(n_chains, chains, links, Lx, Ly, Lz)    
##    file1=open("chain_conn","w")
##    file2=open("links_conn","w")
##    for i in range(0, n_chains): 
##        file1.write("%4i  %4i  %4i  %6.4f\n" % (chains[i,0], chains[i,1], chains[i,2], dist[i,3]))
##    
##    for i in range(0, n_links):
##        file2.write("%4i  %4i  %4i\n" % (conn[i,0], conn[i,1], conn[i,2]))
##    
##    file1.close()
##    file2.close()

    xlo = 0; ylo = 0; zlo = 0
    xhi = L; yhi = L; zhi = L
    atom_types = 2; bond_types = 1; n_break = 0
    mass= np.full(atom_types, 1, dtype = float)
#    ioLAMMPS.writeLAMMPS(xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, n_loops, n_dang, n_free, n_break, links, chains, atom_types, bond_types, mass, loop_atoms)
#    print(len(new_chains[:,0]))
    ioLAMMPS.writeLAMMPSafternetgen('network.txt', xlo, xhi, ylo, yhi, zlo, zhi, links, new_chains, atom_types, bond_types, mass, loop_atoms)

##   [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
##           atom_types, bond_types, mass] = netgen.generate_network(prob, func, N, L, l0, n_chains, n_links)
##
##    n_atoms = n_links
##    n_bonds = len(new_chains[:,0])  
    
#    return xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, links, new_chains, atoms_types, bond_types, mass
#    return links, new_chains, conn, n_loops, n_dang, n_free, n_break
    
#    return loop_atoms
