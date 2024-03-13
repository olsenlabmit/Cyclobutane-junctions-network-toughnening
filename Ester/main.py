#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

"""
#######################################
#                                     #
#-- Fracture Simulation of Networks --#

#Modification for cyclobutane junctions: Devosmita Sen Sept 2023##
#                                     #
#######################################

 Overall Framework (Steps):
     1. Generate a Network following the algorithm published
        by AA Gusev, Macromolecules, 2019, 52, 9, 3244-3251
        if gen_net = 0, then it reads topology from user-supplied 
        network.txt file present in this folder
     
     2. Force relaxtion of network using Fast Inertial Relaxation Engine (FIRE) 
        to obtain the equilibrium positions of crosslinks (min-energy configuration)

     3. Compute Properties: Energy, Gamma (prestretch), and 
        Stress (all 6 componenets) 
     
     4. Deform the network (tensile) in desired direction by 
        strain format by supplying lambda_x, lambda_y, lambda_z

     5. Break bonds using Kintetic Theory of Fracture (force-activated KMC) 
        presently implemented algorithm is ispired by 
        Termonia et al., Macromolecules, 1985, 18, 2246

     6. Repeat steps 2-5 until the given extension (lam_total) is achived OR    
        stress decreases below a certain (user-specified) value 
        indicating that material is completey fractured.
"""

import time
import math
import random
import netgen
import ioLAMMPS
import matplotlib
matplotlib.use('Agg')
import numpy as np
from relax import Optimizer
from numpy import linalg as LA
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import param as p

##random.seed(a=20)
random.seed(a=None, version=2)
print('First random number of this seed: %d'%(random.randint(0, 10000))) 
# This is just to check whether different jobs have different seeds


##def show_exception_and_exit(exc_type, exc_value, tb):
##    import traceback
##    traceback.print_exception(exc_type, exc_value, tb)
####    raw_input("Press key to exit.")
##    sys.exit(-1)
##
##import sys
##sys.excepthook = show_exception_and_exit



netgen_flag = 1
swell = 0
if(netgen_flag==0):

   vflag = 0
   N = 12   
   print('--------------------------')   
   print('----Reading Network-------')   
   print('--------------------------')   
   [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
           atom_types, bond_types, mass, loop_atoms, crosslinker_array, chain_array] = ioLAMMPS.readLAMMPS("network.txt", N, vflag)
   
   print('xlo, xhi',xlo, xhi) 
   print('ylo, yhi',ylo, yhi) 
   print('zlo, zhi',zlo, zhi) 
   print('n_atoms', n_atoms) 
   print('n_bonds', n_bonds) 
   print('atom_types = ', atom_types) 
   print('bond_types = ', bond_types) 
   print('mass = ', mass) 
   print('primary loops = ', len(loop_atoms)) 
   print('--------------------------')   

elif(netgen_flag==1):

   func = p.func
   N    = p.N
   rho  = p.rho
   l0   = p.l0
   prob = p.prob
   n_chains  = p.n_chains
   n_links   = int(2*n_chains/func)
   L = p.L

   netgen.generate_network(prob, func, N, L, l0, n_chains, n_links)

   [xlo, xhi, ylo, yhi, zlo, zhi, n_atoms, n_bonds, atoms, bonds, 
           atom_types, bond_types, mass, loop_atoms, crosslinker_array, chain_array] = ioLAMMPS.readLAMMPS("network.txt", N, 0)
##   stop

else:
   print('Invalid network generation flag')

##stop
fstr=open('stress','w')
fstr.write('#Lx, Ly, Lz, lambda, FE, deltaFE, st[0], st[1], st[2], st[3], st[4], st[5]\n') 

flen=open('strand_lengths','w')
flen.write('#lambda, ave(R), max(R)\n') 

fkmc=open('KMC_stats','w')
fkmc.write('#lambda, init bonds, final bonds\n') 
#-------------------------------------#
#       Simulation Parameters         #
#-------------------------------------#

#N  = 12
N = p.N
b=p.b
Nb=N*b
K  = p.K
r0 = p.r0
##U0  = 1
##tau = 1
del_t = p.del_t
erate = p.erate
lam_max = p.lam_max 
tol = p.tol
max_itr = p.max_itr
write_itr = p.write_itr
wrt_step = p.wrt_step



#-------------------------------------#
#       First Force Relaxation        #
#-------------------------------------#
M=np.zeros(len(bonds[:,0]),dtype='int') # index is the original chain indices, column 1 is the current chain index
        # initially, both the columns contain the same value
for i in range(0,len(bonds[:,0])):
   M[i]=i
   
mymin = Optimizer(atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, N,p.E_b, 'Mao')
[e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, M,chain_array,'log.txt')

if(swell==1):
   ioLAMMPS.writeLAMMPS('restart_network_01.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
   # Swelling the network to V = 2
   scale_x = 1.26
   scale_y = 1.26
   scale_z = 1.26
   mymin.change_box(scale_x, scale_y, scale_z)    
   [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr, M,chain_array,'log.txt')

ioLAMMPS.writeLAMMPS('restart_network_0.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

dist = mymin.bondlengths()
Lx0 = mymin.xhi-mymin.xlo
BE0 = e
[pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(M,chain_array)
fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                          %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                           (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
fstr.flush()

flen.write('%7.4f  %7.4f  %7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
flen.flush()

fkmc.write('%7.4f  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds, n_bonds))
fkmc.flush()

ioLAMMPS.writeLAMMPS('restart_network_0.txt', mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, 
                                  mymin.zhi, mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)
##stop
#-------------------------------------#
# Tensile deformation: lambda/scales  #
#-------------------------------------#
steps = int((lam_max-1)/(erate*del_t))
print('Deformation steps = ',steps)
begin_break = -1         # -1 implies that bond breaking begins right from start
#begin_break = n_steps   # implies bond breaking will begin after n_steps of deformation



   
##stop

num_broken_cumulative = np.zeros(4)
num_considered_cumulative = np.zeros(4)
rates=np.zeros(4)
plt.figure()
n_i_arr=[]
num_broken_in_delta_t_step_arr=[]
delta_t_iteration_arr=[]
num_broken_cumulative_arr_0=[]
num_broken_cumulative_arr_1=[]
num_broken_cumulative_arr_2=[]
num_broken_cumulative_arr_3=[]

num_considered_cumulative_arr_0=[]
num_considered_cumulative_arr_1=[]
num_considered_cumulative_arr_2=[]
num_considered_cumulative_arr_3=[]

sum_ni_arr = []
##norm_ratesa
norm_rates_0_arr=[]
norm_rates_1_arr=[]
norm_rates_2_arr=[]
norm_rates_3_arr=[]
   
for i in range(0,steps):

    scale_x = (1+(i+1)*erate*del_t)/(1+i*erate*del_t)
    scale_y = scale_z = 1.0/math.sqrt(scale_x)
    mymin.change_box(scale_x, scale_y, scale_z)    
    [e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr,M,chain_array, 'log.txt')
    [pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(M,chain_array)
    fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                     %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                                  (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
    fstr.flush()

    dist = mymin.bondlengths()
    flen.write('%7.4f  %7.4f  %7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
    flen.flush()
   
    if((i+1)%wrt_step==0): 
      filename = 'restart_network_%d.txt' %(i+1)
      ioLAMMPS.writeLAMMPS(filename, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                           mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

    if(i > begin_break):
      # U0, tau, del_t, pflag, index
##      print('num_broken_cumulative',num_broken_cumulative)
      num_broken_cumulative_old=num_broken_cumulative.copy()
      [t, n_bonds_init, n_bonds_final,M,chain_array, crosslinker_array, num_broken, num_broken_cumulative_new,num_considered_cumulative, sum_ni,[norm_rates_0,norm_rates_1,norm_rates_2,norm_rates_3]] = mymin.KMCbondbreak([p.U0,p.U2],[p.tau0,p.tau2],[p.delta0,p.delta2], p.del_t, 0, i+1,M,chain_array, crosslinker_array,num_broken_cumulative,num_considered_cumulative)
##      print('num_broken_cumulative',num_broken_cumulative)
      num_broken_in_delta_t_step_arr.append(num_broken_cumulative_new-num_broken_cumulative_old)
      norm_rates_0_arr.append(norm_rates_0)
      norm_rates_1_arr.append(norm_rates_1)
      norm_rates_2_arr.append(norm_rates_2)
      norm_rates_3_arr.append(norm_rates_3)

      
##      stop
      
      delta_t_iteration_arr.append(i)
      
      num_broken_cumulative=num_broken_cumulative_new
      n_i_arr_temp=[]
####      for chain in chain_array:
####         n_i_arr_temp.append(chain.n)
######      print('num_broken_cumulative',num_broken_cumulative)
######      stop
####      n_i_arr.append(np.array(n_i_arr_temp))
##      print('n_i_arr_temp',n_i_arr_temp)
      
####      stop
##      plt.plot(i,num_broken_cumulative[0],marker="o",color="r")#,label="PEG")
##      plt.plot(i,num_broken_cumulative[1],marker="o",color="g")#,label="Broken Crosslinkers")
##      plt.plot(i,num_broken_cumulative[2],marker="o",color="b")#,label="1,2 regiochemistry")
##      plt.plot(i,num_broken_cumulative[3],marker="o",color="m")#,label="1,3 regiochemistry")
##
##      plt.plot(i,num_considered_cumulative[0],marker="_",color="r")#,label="PEG")
##      plt.plot(i,num_considered_cumulative[1],marker="_",color="g")#,label="Broken Crosslinkers")
##      plt.plot(i,num_considered_cumulative[2],marker="_",color="b")#,label="1,2 regiochemistry")
##      plt.plot(i,num_considered_cumulative[3],marker="_",color="m")#,label="1,3 regiochemistry")
      num_considered_cumulative_arr_0.append(num_considered_cumulative[0])
      num_considered_cumulative_arr_1.append(num_considered_cumulative[1])
      num_considered_cumulative_arr_2.append(num_considered_cumulative[2])
      num_considered_cumulative_arr_3.append(num_considered_cumulative[3])

      num_broken_cumulative_arr_0.append(num_broken_cumulative[0])
      num_broken_cumulative_arr_1.append(num_broken_cumulative[1])
      num_broken_cumulative_arr_2.append(num_broken_cumulative[2])
      num_broken_cumulative_arr_3.append(num_broken_cumulative[3])

      sum_ni_arr.append(sum_ni)
      
##      plt.xlabel('Iteration')
##      plt.ylabel('Bonds Considered')
#      plt.legend()
  #    plt.title('Bonds Broken in each Iteration')
      
   

      
##      print('IN MAIN:')
##      print('############################')
##      print('############################')
##      print('crosslinker_array[199].cls',crosslinker_array[199].cls)
##      print('crosslinker_array[199].chains',crosslinker_array[199].chains)
##      print('############################')
##      print('############################')
      fkmc.write('%7.4f  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final))
      
      fkmc.flush()
##      print('############################')
##      print('NUM_BROKEN',num_broken)
##      print('############################')


plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_cumulative_arr_0),'ro')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_cumulative_arr_1),'go')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_cumulative_arr_2),'bo')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_cumulative_arr_3),'mo')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_considered_cumulative_arr_0),'r-')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_considered_cumulative_arr_1),'g-')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_considered_cumulative_arr_2),'b-')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_considered_cumulative_arr_3),'m-')

plt.title('Bonds Broken and considered cumulative')
plt.legend(['PEG','broken crosslinker in PEG','1,2 regiochem','1,3 regiochem'])
plt.savefig('bonds_broken_cumulative')
np.savetxt("bonds_broken_cumulative.txt",np.transpose(np.array([delta_t_iteration_arr,num_broken_cumulative_arr_0, num_broken_cumulative_arr_1, num_broken_cumulative_arr_2, num_broken_cumulative_arr_3, num_considered_cumulative_arr_0,num_considered_cumulative_arr_1,num_considered_cumulative_arr_2,num_considered_cumulative_arr_3])), header='delta_i PEG(broken) broken_chain_in_PEG(broken) 1,2_regio(broken) 1,3 regio(broken) PEG(considered) broken_chain_in_PEG(considered) 1,2_regio(considered) 1,3 regio(considered) ')

plt.figure()
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_in_delta_t_step_arr)[:,0],'ro')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_in_delta_t_step_arr)[:,1],'go')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_in_delta_t_step_arr)[:,2],'bo')
plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_in_delta_t_step_arr)[:,3],'mo')
plt.legend(['PEG','broken crosslinker in PEG','1,2 regiochem','1,3 regiochem'])
plt.title('Bonds Broken in each Iteration')
plt.savefig('bonds_broken_step')
np.savetxt("bonds_broken_step.txt",np.transpose(np.array([delta_t_iteration_arr,np.array(num_broken_in_delta_t_step_arr)[:,0], np.array(num_broken_in_delta_t_step_arr)[:,1], np.array(num_broken_in_delta_t_step_arr)[:,2], np.array(num_broken_in_delta_t_step_arr)[:,3]])), header='delta_i PEG broken_chain_in_PEG 1,2_regio 1,3 regio')


plt.figure()
plt.plot(np.array(delta_t_iteration_arr),np.array(norm_rates_0_arr),'ro-')
plt.plot(np.array(delta_t_iteration_arr),np.array(norm_rates_1_arr),'go-')
plt.plot(np.array(delta_t_iteration_arr),np.array(norm_rates_2_arr),'bo-')
plt.plot(np.array(delta_t_iteration_arr),np.array(norm_rates_3_arr),'mo-')
plt.legend(['PEG','broken crosslinker in PEG','1,2 regiochem','1,3 regiochem'])
plt.title('Rates each Iteration')
plt.savefig('rates_ite')
np.savetxt("rates_ite.txt",np.transpose(np.array([delta_t_iteration_arr,norm_rates_0_arr, norm_rates_1_arr, norm_rates_2_arr, norm_rates_3_arr])), header='delta_i PEG broken_chain_in_PEG 1,2_regio 1,3 regio')



plt.figure()
####print(np.shape(n_i_arr))
##n_i_arr=np.array(n_i_arr)
##for j in n_i_arr:
##   plt.plot(np.arange(0,np.shape(j)[0]),j)

plt.plot(np.array(delta_t_iteration_arr),sum_ni_arr)
plt.title('sum_n_i all chains- variation with iteration')
plt.savefig('sum_n_i_with_iteration')
np.savetxt("sum_n_i_ite.txt",np.transpose(np.array([delta_t_iteration_arr,sum_ni_arr])), header='delta_i PEG broken_chain_in_PEG 1,2_regio 1,3 regio')




#---------------------------------#
#     Final Network Properties    #
#---------------------------------#

  
[e, Gamma] = mymin.fire_iterate(tol, max_itr, write_itr,M,chain_array, 'log.txt')
[pxx, pyy, pzz, pxy, pyz, pzx] = mymin.compute_pressure(M,chain_array)
fstr.write('%7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f\n' \
                                 %(mymin.xhi-mymin.xlo, mymin.yhi-mymin.ylo, mymin.zhi-mymin.zlo, 
                              (mymin.xhi-mymin.xlo)/Lx0, e, e-BE0, pxx, pyy, pzz, pxy, pyz, pzx)) 
fstr.flush()

dist = mymin.bondlengths()
flen.write('%7.4f  %7.4f  %7.4f\n'%((mymin.xhi-mymin.xlo)/Lx0, np.mean(dist[:,3])/N, np.max(dist[:,3])/N))
flen.flush()

fkmc.write('%7.4f  %5i  %5i\n'%((mymin.xhi-mymin.xlo)/Lx0, n_bonds_init, n_bonds_final))
fkmc.flush()

filename = 'restart_network_%d.txt' %(i+1)
ioLAMMPS.writeLAMMPS(filename, mymin.xlo, mymin.xhi, mymin.ylo, mymin.yhi, mymin.zlo, mymin.zhi,
                                       mymin.atoms, mymin.bonds, atom_types, bond_types, mass, loop_atoms)

fstr.close()
flen.close()
fkmc.close()

##plt.show()
##plt.plot(np.array(delta_t_iteration_arr),np.array(num_broken_in_delta_t_step_arr)[])
