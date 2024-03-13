#!/use/local/bin/env python
# -*- coding: utf-8 -*-
##
##-------------------------------------------------
## Fast Inertial Relaxation Engine (FIRE) Optimizer
## Ref: Bitzek et al, PRL, 97, 170201 (2006)
##
## Author: Akash Arora
## Implementation is inspired from LAMMPS and ASE Master Code
##-------------------------------------------------

import math
import time
import random
import numpy as np
import scipy.optimize as opt
from numpy import linalg as LA
from scipy.optimize import fsolve

class crosslinker:
   def __init__(self,index):
        self.index = index
        # initially all chains and crossliners set to -1 because nothing is connected initially
##        self.chain_1=-1  # chain connected to this crosslinker via bond 1
##        self.chain_2=-1
##        self.chain_3=-1
##        self.chain_4=-1
##        self.cl_1=-1  # crosslinker connected to this crosslinker via bond 1 (nd chain_1)
##        self.cl_2=-1
##        self.cl_3=-1
##        self.cl_4=-1
        self.chains=[] # list of connected chains
        self.cls=[] # list of connected crosslinkers corresponding to the chains 

class chain:
   def __init__(self,index,cl_1,cl_2,n=0):
        self.index = index
        # initially all crossliners set to -1 because nothing is connected initially
        self.cl_1=cl_1  # crosslinker connected to first end of this chain 
        self.cl_2=cl_2
        self.n=n # number of broken junction pieces included in this chain
        
class Optimizer(object):

    def __init__(self, atoms, bonds, xlo, xhi, ylo, yhi, zlo, zhi, K, r0, N, E_b, ftype):

        self.atoms = atoms
        self.bonds = bonds
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi
        self.K = K
        self.r0 = r0          
        self.N = N          
        self.ftype = ftype
        self.E_b=E_b
##        self.M=M

    def bondlengths(self):
     
        atoms = self.atoms
        bonds = self.bonds
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        n_atoms = len(self.atoms[:,0])
        n_bonds = len(self.bonds[:,0])

        dist = np.zeros((n_bonds,4), dtype=float)

        for i in range (0, n_bonds):
          
              lnk_1 = bonds[i,2]-1
              lnk_2 = bonds[i,3]-1
              delr = atoms[lnk_1,:] - atoms[lnk_2,:]
              
              delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
              delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
              delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
                   
              dist[i,0:3] = delr
              dist[i,3] = LA.norm(delr)
    
        return dist

    
    def invlangevin(self, x):
        return x*(2.99942 - 2.57332*x + 0.654805*x**2)/(1-0.894936*x - 0.105064*x**2)

    def kuhn_stretch(self, lam, E_b):
       
        def func(x, lam, E_b):
            y = lam/x
            beta = self.invlangevin(y)
            return E_b*np.log(x) - lam*beta/x
   
        if lam == 0:
           return 1
        else:
           lam_b = opt.root_scalar(func,args=(lam, E_b),bracket=[lam,lam+1],x0=lam+0.05)
           return lam_b.root

    def get_bondforce(self, r,n_i):

        K  = self.K
        r0 = self.r0
        Nb = self.N*(n_i+1) # b = 1 (lenght scale of the system)
        E_b = self.E_b
 
        x = (r-r0)/Nb
        if(x<0.90):
           lam_b = 1.0
           fbkT  = self.invlangevin(x)
           fbond = -K*fbkT/r
        elif(x<1.4):
           lam_b = self.kuhn_stretch(x, E_b)
           fbkT  = self.invlangevin(x/lam_b)/lam_b
           fbond = -K*fbkT/r
        else:
           lam_b = x + 0.05
           fbkT  = 325 + 400*(x-1.4)            
           fbond = -K*fbkT/r
 
        return fbond, lam_b  
          

    def get_force(self,M,chain_array):
       
        N = self.N
        E_b = self.E_b
        atoms = self.atoms
        bonds = self.bonds
        ftype = self.ftype
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        n_atoms = len(atoms[:,0])
        n_bonds = len(bonds[:,0])
       
        e = 0.0 
        Gamma = 0.0
        f =  np.zeros((n_atoms,3), dtype = float)
        for i in range(0, n_bonds):
           
            i_orig=np.where(M==i)[0][0]
            n_i=chain_array[i_orig].n
            
            lnk_1 = bonds[i,2]-1
            lnk_2 = bonds[i,3]-1
            delr = atoms[lnk_1,:] - atoms[lnk_2,:]
            
            delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
            delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
            delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
                 
            r = LA.norm(delr)
            if (r > 0): 
               [fbond, lam_b] = self.get_bondforce(r,n_i) 
               lam = (r-self.r0)/N
               beta = -fbond*r/self.K*lam_b
               e_bond = N*0.5*E_b*math.log(lam_b)**2
##               if(beta==0):
##                  print('fbond',fbond)
##                  print('lam_b',lam_b)
##                  print('r',r)
               try:
                  e_stretch = N*( (lam/lam_b)*beta + math.log(beta/math.sinh(beta)))
               except OverflowError:
                  print('fbond',fbond)
                  print('lam_b',lam_b)
                  print('r',r)
                  print('beta',beta)
                  print('i',i)
                  print(lnk_1+1)
                  print(lnk_2+1)
##                  print(math.sinh(beta))
##                  print(beta/math.sinh(beta))
##                  print(math.log(beta/math.sinh(beta)))
                  stop
               e = e + e_bond + e_stretch
            else:
               fbond = 0.0
               e = e + 0.0
       
            Gamma = Gamma + r*r
       
            # apply force to each of 2 atoms        
            if (lnk_1 < n_atoms):
               f[lnk_1,0] = f[lnk_1,0] + delr[0]*fbond
               f[lnk_1,1] = f[lnk_1,1] + delr[1]*fbond
               f[lnk_1,2] = f[lnk_1,2] + delr[2]*fbond
        
            if (lnk_2 < n_atoms):
               f[lnk_2,0] = f[lnk_2,0] - delr[0]*fbond
               f[lnk_2,1] = f[lnk_2,1] - delr[1]*fbond
               f[lnk_2,2] = f[lnk_2,2] - delr[2]*fbond
        
        return f, e, Gamma


  
    def get_force_cl_chain(self,M,chain_array): # get matrix of all force vector on cl_idx due to the tension in chain_idx
       
        N = self.N
        E_b = self.E_b
        atoms = self.atoms
        bonds = self.bonds
        ftype = self.ftype
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        n_atoms = len(atoms[:,0])
        n_bonds = len(bonds[:,0])
        
        e = 0.0 
        Gamma = 0.0
        f =  np.zeros((n_atoms,n_bonds,3), dtype = float) #f[cl,ch,:]=x,y,z components of force on crosslinker cl due to chain ch
        
        for i in range(0, n_bonds):

##            i=chain_idx
            i_orig=np.where(M==i)[0][0]
            n_i=chain_array[i_orig].n
            lnk_1 = bonds[i,2]-1
            lnk_2 = bonds[i,3]-1
            delr = atoms[lnk_1,:] - atoms[lnk_2,:]
            
            delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
            delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
            delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
                 
            r = LA.norm(delr)
            if (r > 0): 
               [fbond, lam_b] = self.get_bondforce(r,n_i) 
               lam = (r-self.r0)/N
               beta = -fbond*r/self.K*lam_b
               e_bond = N*0.5*(E_b/(n_i+1))*math.log(lam_b)**2
               e_stretch = N*( (lam/lam_b)*beta + math.log(beta/math.sinh(beta)))
               e = e + e_bond + e_stretch
            else:
               fbond = 0.0
               e = e + 0.0
       
            Gamma = Gamma + r*r
       
            # apply force to each of 2 atoms        
            if (lnk_1 < n_atoms):
               f[lnk_1,i,0] = f[lnk_1,i,0] + delr[0]*fbond
               f[lnk_1,i,1] = f[lnk_1,i,1] + delr[1]*fbond
               f[lnk_1,i,2] = f[lnk_1,i,2] + delr[2]*fbond
        
            if (lnk_2 < n_atoms):
               f[lnk_2,i,0] = f[lnk_2,i,0] - delr[0]*fbond
               f[lnk_2,i,1] = f[lnk_2,i,1] - delr[1]*fbond
               f[lnk_2,i,2] = f[lnk_2,i,2] - delr[2]*fbond
##        print(f[0,0,:])
##        print(f)
        return f

    
 
    def fire_iterate(self, ftol, maxiter, write_itr, M,chain_array,logfilename):

        tstart = time.time()

        ## Optimization parameters:
        eps_energy = 1.0e-8
        delaystep = 5
        dt_grow = 1.1
        dt_shrink = 0.5
        alpha0 = 0.1
        alpha_shrink = 0.99
        tmax = 10.0
        maxmove = 0.1
        last_negative = 0

        dt = 0.005
        dtmax = dt*tmax
        alpha = alpha0
        last_negative = 0       
 
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        n_atoms = len(self.atoms[:,0])
        n_bonds = len(self.bonds[:,0])
        v = np.zeros((n_atoms,3), dtype = float)

        n_bonds = len(self.bonds)
        dist = np.zeros((n_bonds,4), dtype=float)


        [f,e,Gamma] = self.get_force(M,chain_array)
        dist = self.bondlengths()

 
        fmaxitr = np.max(np.max(np.absolute(f)))
        fnormitr = math.sqrt(np.vdot(f,f))
##        logfile = open(logfilename,'w') 
##        logfile.write('FIRE: iter  Energy  fmax  fnorm  avg(r)/Nb  max(r)/Nb\n')
##        logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
##                              ('FIRE', 0, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
##        logfile.flush()
####        print('FIRE: iter  Energy  fmax  fnorm  avg(r)/Nb  max(r)/Nb')
####        print('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f' %
####                              ('FIRE', 0, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))

        for itr in range (0, maxiter):
         
          vdotf = np.vdot(v,f) 
          if (vdotf > 0.0):
             vdotv = np.vdot(v,v)
             fdotf = np.vdot(f,f) 
             scale1 = 1.0 - alpha
             if (fdotf == 0.0): scale2 = 0.0
             else: scale2 = alpha * math.sqrt(vdotv/fdotf)
             v = scale1*v + scale2*f
              
             if (itr - last_negative > delaystep):
                 dt = min(dt*dt_grow,dtmax)
                 alpha = alpha*alpha_shrink
      
          else:
             last_negative = itr
             dt = dt*dt_shrink
             alpha = alpha0
             v[:] = v[:]*0.0
      
          v = v + dt*f 
          dr = dt*v
          normdr = np.sqrt(np.vdot(dr, dr))
          if (normdr > maxmove):
              dr = maxmove * dr / normdr

          self.atoms = self.atoms + dr
          for i in range(0, n_atoms):
              self.atoms[i,0] = self.atoms[i,0] - math.floor((self.atoms[i,0]-self.xlo)/Lx)*Lx
              self.atoms[i,1] = self.atoms[i,1] - math.floor((self.atoms[i,1]-self.ylo)/Ly)*Ly
              self.atoms[i,2] = self.atoms[i,2] - math.floor((self.atoms[i,2]-self.zlo)/Lz)*Lz
          
  
          [f,e,Gamma] = self.get_force(M,chain_array)
          fmaxitr = np.max(np.max(np.absolute(f)))
          fnormitr = math.sqrt(np.vdot(f,f))


          if((itr+1)%write_itr==0):
             continue
####             dist = self.bondlengths()
##             logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
##                                  ('FIRE', itr+1, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
##             logfile.flush()

             # Print on screen
####             print('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f' %
####                               ('FIRE', itr+1,  e, fmaxitr, fnormitr,  np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
        
   
          # Checking for convergence
          if (fnormitr < ftol):
             dist = self.bondlengths()
             tend = time.time()
##             logfile.write('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f\n' %
##                                  ('FIRE', itr+1, e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
##             logfile.flush()
####             print('%s: %5d  %9.6f  %9.6f  %9.6f  %9.4f  %9.4f' %
####                               ('FIRE', itr+1,  e, fmaxitr, fnormitr, np.mean(dist[:,3])/self.N, np.max(dist[:,3])/self.N))
####             print('Iterations converged, Time taken: %7.4f' %(tend-tstart))
             break
          elif (itr == maxiter-1):
             print('Maximum iterations reached')
     

##        logfile.close() 
        
        return e, Gamma       
                

    def compute_pressure(self,M,chain_array):

        K = self.K
        r0 = self.r0
        ftype = self.ftype
        Lx = self.xhi - self.xlo
        Ly = self.yhi - self.ylo
        Lz = self.zhi - self.zlo
        atoms = self.atoms
        bonds = self.bonds
        n_atoms = len(atoms[:,0])
        n_bonds = len(bonds[:,0])
       
        pxx = pyy = pzz = pxy = pyz = pzx = 0.0
        sigma = np.zeros((n_atoms,6), dtype=float)
        inv_volume = 1.0/(Lx*Ly*Lz)
        for i in range(0, n_bonds):
            i_orig=np.where(M==i)[0][0]
            n_i=chain_array[i_orig].n
            lnk_1 = bonds[i,2]-1
            lnk_2 = bonds[i,3]-1
            delr = atoms[lnk_1,:] - atoms[lnk_2,:]
            
            delr[0] = delr[0] - int(round(delr[0]/Lx))*Lx
            delr[1] = delr[1] - int(round(delr[1]/Ly))*Ly
            delr[2] = delr[2] - int(round(delr[2]/Lz))*Lz
                 
            r = LA.norm(delr)
            if (r > 0.0):
               if(ftype=='Mao'): [fbond, lam_b] = self.get_bondforce(r,n_i)
               else: fbond = self.get_bondforce(r) 
            else: fbond = 0.0
            
            # apply pressure to each of the 2 atoms   
            # And for each of the 6 components     
            if (lnk_1 < n_atoms):
               sigma[lnk_1,0] = sigma[lnk_1,0] + 0.5*delr[0]*delr[0]*fbond
               sigma[lnk_1,1] = sigma[lnk_1,1] + 0.5*delr[1]*delr[1]*fbond
               sigma[lnk_1,2] = sigma[lnk_1,2] + 0.5*delr[2]*delr[2]*fbond
               sigma[lnk_1,3] = sigma[lnk_1,3] + 0.5*delr[0]*delr[1]*fbond
               sigma[lnk_1,4] = sigma[lnk_1,4] + 0.5*delr[1]*delr[2]*fbond
               sigma[lnk_1,5] = sigma[lnk_1,5] + 0.5*delr[2]*delr[0]*fbond
        
            if (lnk_2 < n_atoms):
               sigma[lnk_2,0] = sigma[lnk_2,0] + 0.5*delr[0]*delr[0]*fbond
               sigma[lnk_2,1] = sigma[lnk_2,1] + 0.5*delr[1]*delr[1]*fbond
               sigma[lnk_2,2] = sigma[lnk_2,2] + 0.5*delr[2]*delr[2]*fbond
               sigma[lnk_2,3] = sigma[lnk_2,3] + 0.5*delr[0]*delr[1]*fbond
               sigma[lnk_2,4] = sigma[lnk_2,4] + 0.5*delr[1]*delr[2]*fbond
               sigma[lnk_2,5] = sigma[lnk_2,5] + 0.5*delr[2]*delr[0]*fbond


        pxx = np.sum(sigma[:,0])*inv_volume
        pyy = np.sum(sigma[:,1])*inv_volume
        pzz = np.sum(sigma[:,2])*inv_volume
        pxy = np.sum(sigma[:,3])*inv_volume
        pyz = np.sum(sigma[:,4])*inv_volume
        pzx = np.sum(sigma[:,5])*inv_volume

        return pxx, pyy, pzz, pxy, pyz, pzx


    def change_box(self, scale_x, scale_y, scale_z):

        xlo = self.xlo
        xhi = self.xhi
        ylo = self.ylo
        yhi = self.yhi
        zlo = self.zlo
        zhi = self.zhi
        atoms = self.atoms
        bonds = self.bonds
        n_atoms = len(atoms[:,0])
        n_bonds = len(bonds[:,0])

        xmid = (xlo+xhi)/2  
        ymid = (ylo+yhi)/2  
        zmid = (zlo+zhi)/2  

        new_xlo = xmid + scale_x*(xlo-xmid)
        new_ylo = ymid + scale_y*(ylo-ymid)
        new_zlo = zmid + scale_z*(zlo-zmid)

        new_xhi = xmid + scale_x*(xhi-xmid)
        new_yhi = ymid + scale_y*(yhi-ymid)
        new_zhi = zmid + scale_z*(zhi-zmid)
        
        newLx = new_xhi - new_xlo
        newLy = new_yhi - new_ylo
        newLz = new_zhi - new_zlo
        for i in range(0, n_atoms):            
            atoms[i,0] = xmid + scale_x*(atoms[i,0]-xmid)
            atoms[i,1] = ymid + scale_y*(atoms[i,1]-ymid)
            atoms[i,2] = zmid + scale_z*(atoms[i,2]-zmid)

        self.atoms = atoms
        self.xlo = new_xlo
        self.xhi = new_xhi
        self.ylo = new_ylo
        self.yhi = new_yhi
        self.zlo = new_zlo
        self.zhi = new_zhi





    def KMCbondbreak(self, U_arr, tau_arr, delta_arr, delta_t, pflag, index_orig,M,chain_array, crosslinker_array,num_broken_cumulative, num_considered_cumulative):

        
##        print('index',index)
        # matrix M maps from the original indices to current indices of chains
        [delta0,delta11,delta12,delta_broken]=delta_arr
        [tau0,tau1,tau_broken]=tau_arr
        [U0,U11,U12,U_broken]=U_arr
        # Material parameters:
        # beta = 1.0 -- All material params, U0 and sigma, are in units of kT. 
        # Main array: Bonds_register = [Activity index, type, index, link1, link2, dist, rate(ri)]
        # All are active at the start (active = 1, break = 0)

        num_broken=np.zeros(4) # 6 types of breaking
        norm_rates_0=0
        norm_rates_1=0
        norm_rates_2=0
        norm_rates_3=0
   
        def get_link_bonds(link, bonds_register):
        
            conn = {}
            a1 = np.where(bonds_register[:,3]==link)
            a2 = np.where(bonds_register[:,4]==link)
            a = np.concatenate((a1[0],a2[0]))
            a = np.unique(a)
            for i in range(0,len(a)):
                if(bonds_register[a[i],0]==1): 
                  conn.update({a[i] : bonds_register[a[i],5]})
           
            conn = dict(sorted(conn.items(), key=lambda x: x[1]))     

            return conn


        Nb = self.N
        ftype = self.ftype
        n_bonds = len(self.bonds[:,0])
        bonds_register = np.zeros((n_bonds,7))
        broken_cl_bonds_register= np.zeros(n_bonds,dtype='float') # contains only the rates corresponding to the bonds of broken junction present in the chains
        # since the topology will be the same, I am only storing the rates
        # the topological changes will be the same as that if the actual chain
        # also- the rates will be equal for all broken crosslinker pieces in a particular chain, hence, I am storing only one value
        # the number of such broken crosslinker bonds will be included while calculating the probability
        #the indices of broken_cl_bonds_register correspond to the current chain indices, not the original indices!
        bonds_register[:,0] = 1   
        bonds_register[:,1:5] = self.bonds
        dist = self.bondlengths()
        bonds_register[:,5] = dist[:,3]
        n_atoms=len(self.atoms[:,0])
        
        register_cl_12_1=np.zeros((n_atoms,9),dtype='float') # first column-force on crosslinker following 1,2 regiochemistry, and possibility 1
        # resulting topologies indicated in the last 4 columns
        # last 4 entries are the resulting crosslinker connections: c1-c2, c3-c4
        register_cl_12_2=np.zeros((n_atoms,9),dtype='float') # 1,2 regiochemistry, type 2
##        register_cl_13_1=np.zeros((n_atoms,3),dtype='float') # 1,3 regiochemistry, bond_1
##        register_cl_13_2=np.zeros((n_atoms,3),dtype='float') # 1,3 regiochemistry
##        register_cl_13_3=np.zeros((n_atoms,3),dtype='float') # 1,3 regiochemistry
##        register_cl_13_4=np.zeros((n_atoms,3),dtype='float') # 1,3 regiochemistry
        # here the last column indicates the bond to be removed 
        f=self.get_force_cl_chain(M,chain_array)
##        print(type(f))
##        print(f[0,:,:])
        for i in range(0,len(self.atoms[:,0])): # crosslinker number
            idx=i
##            print(idx)
##            stop
##            print(crosslinker_array[idx].cls)
##            print(crosslinker_array[idx].chains)
##            print(len(crosslinker_array[idx].chains))
##            print('len(crosslinker_array[idx].chains)',len(crosslinker_array[idx].chains))
            if(len(crosslinker_array[idx].chains)==0): # unreacted crosslinker
                ch_1=-1
                ch_2=-1
                ch_3=-1
                ch_4=-1
                cl_1=-1
                cl_2=-1
                cl_3=-1
                cl_4=-1
            elif(len(crosslinker_array[idx].chains)==1): # unreacted crosslinker
                k = random.randint(0, 3)
                if(k==0):
                    [ch_1]=crosslinker_array[idx].chains
                    ch_2=-1
                    ch_3=-1
                    ch_4=-1
                    [cl_1]=crosslinker_array[idx].cls
                    cl_2=-1
                    cl_3=-1
                    cl_4=-1
                elif(k==1):
                    [ch_2]=crosslinker_array[idx].chains
                    ch_1=-1
                    ch_3=-1
                    ch_4=-1
                    [cl_2]=crosslinker_array[idx].cls
                    cl_1=-1
                    cl_3=-1
                    cl_4=-1
                   
                elif(k==2):
                    [ch_3]=crosslinker_array[idx].chains
                    ch_2=-1
                    ch_1=-1
                    ch_4=-1
                    [cl_3]=crosslinker_array[idx].cls
                    cl_2=-1
                    cl_1=-1
                    cl_4=-1
                elif(k==3):
                    [ch_4]=crosslinker_array[idx].chains
                    ch_2=-1
                    ch_3=-1
                    ch_1=-1
                    [cl_4]=crosslinker_array[idx].cls
                    cl_2=-1
                    cl_3=-1
                    cl_1=-1
            elif(len(crosslinker_array[idx].chains)==2): # primary loop
                k = random.randint(0, 5)
##                print('k',k)
                if(k==0):
                    [ch_1,ch_2]=crosslinker_array[idx].chains
                    ch_3=-1
                    ch_4=-1
                    [cl_1,cl_2]=crosslinker_array[idx].cls
                    cl_3=-1
                    cl_4=-1
                elif(k==1):
                    [ch_1,ch_3]=crosslinker_array[idx].chains
                    ch_2=-1
                    ch_4=-1
                    [cl_1,cl_3]=crosslinker_array[idx].cls
                    cl_2=-1
                    cl_4=-1
                elif(k==2):
                    [ch_1,ch_4]=crosslinker_array[idx].chains
                    ch_2=-1
                    ch_3=-1
                    [cl_1,cl_4]=crosslinker_array[idx].cls
                    cl_2=-1
                    cl_3=-1
                elif(k==3):
                    [ch_2,ch_3]=crosslinker_array[idx].chains
                    ch_1=-1
                    ch_4=-1
                    [cl_2,cl_3]=crosslinker_array[idx].cls
                    cl_1=-1
                    cl_4=-1
##                    print('cl_1',cl_1)
                elif(k==4):
                    [ch_2,ch_4]=crosslinker_array[idx].chains
                    ch_1=-1
                    ch_3=-1
                    [cl_2,cl_4]=crosslinker_array[idx].cls
                    cl_1=-1
                    cl_3=-1
                elif(k==5):
                    [ch_3,ch_4]=crosslinker_array[idx].chains
                    ch_1=-1
                    ch_2=-1
                    [cl_3,cl_4]=crosslinker_array[idx].cls
                    cl_1=-1
                    cl_2=-1
            elif(len(crosslinker_array[idx].chains)==3): # unreacted crosslinker
                k = random.randint(0, 3)
                if(k==0):
                    [ch_1,ch_2,ch_3]=crosslinker_array[idx].chains
                    ch_4=-1
                    [cl_1,cl_2,ch_3]=crosslinker_array[idx].cls
                    cl_4=-1
                elif(k==1):
                    [ch_1,ch_2,ch_4]=crosslinker_array[idx].chains
                    ch_3=-1
                   
                    [cl_1,cl_2,cl_4]=crosslinker_array[idx].cls
                    cl_3=-1
                   
                elif(k==2):
                    [ch_1,ch_3,ch_4]=crosslinker_array[idx].chains
                    ch_2=-1
                   
                    [cl_1,cl_3,cl_4]=crosslinker_array[idx].cls
                    cl_2=-1
                elif(k==3):
                    [ch_2,ch_3,ch_4]=crosslinker_array[idx].chains
                    ch_1=-1
                   
                    [cl_2,cl_3,cl_4]=crosslinker_array[idx].cls
                    cl_1=-1
                
                
##            
            else:
                [ch_1,ch_2,ch_3,ch_4]=crosslinker_array[idx].chains
                [cl_1,cl_2,cl_3,cl_4]=crosslinker_array[idx].cls
##            if(cl_4==20):
##               stop
##
##            if(idx==20):
##               print(crosslinker_array[idx].chains)
##               print(crosslinker_array[idx].cls)
##               stop
##                print(crosslinker_array[idx].cls)
##                print(crosslinker_array[idx].chains)
##            print([ch_1,ch_2,ch_3,ch_4])
##            print([cl_1,cl_2,cl_3,cl_4])
            # map to current indices
##            print('[cl_1,cl_2,cl_3,cl_4]',[cl_1,cl_2,cl_3,cl_4])
##            print('[ch_1,ch_2,ch_3,ch_4]',[ch_1,ch_2,ch_3,ch_4])
            try:
               if(ch_1!=-1 and ch_2!=-1 and ch_3!=-1 and ch_4!=-1):
                   ch_1_curr=M[ch_1]# find current index
                   ch_2_curr=M[ch_2]
                   ch_3_curr=M[ch_3]
                   ch_4_curr=M[ch_4]
                   f1=f[idx,ch_1_curr,:]
                   f2=f[idx,ch_2_curr,:]
                   f3=f[idx,ch_3_curr,:]
                   f4=f[idx,ch_4_curr,:]

               elif(ch_1==-1):
                   ch_1_curr=-1 #M[ch_1]# find current index
                   f1=0 #f[idx,ch_1_curr,:]

   ##                ch_1_curr=M[ch_1]# find current index
                   ch_2_curr=M[ch_2]
                   ch_3_curr=M[ch_3]
                   ch_4_curr=M[ch_4]
   ##                f1=f[idx,ch_1_curr,:]
                   f2=f[idx,ch_2_curr,:]
                   f3=f[idx,ch_3_curr,:]
                   f4=f[idx,ch_4_curr,:]
               elif(ch_2==-1):
                   ch_2_curr=-1 #M[ch_2]
                   f2=0 #f[idx,ch_2_curr,:]

                   ch_1_curr=M[ch_1]# find current index
   ##                ch_2_curr=M[ch_2]
                   ch_3_curr=M[ch_3]
                   ch_4_curr=M[ch_4]
                   f1=f[idx,ch_1_curr,:]
   ##                f2=f[idx,ch_2_curr,:]
                   f3=f[idx,ch_3_curr,:]
                   f4=f[idx,ch_4_curr,:]
               elif(ch_3==-1):
                   ch_3_curr=-1 #M[ch_3]
                   f3=0#f[idx,ch_3_curr,:]

                   ch_1_curr=M[ch_1]# find current index
                   ch_2_curr=M[ch_2]
   ##                ch_3_curr=M[ch_3]
                   ch_4_curr=M[ch_4]
                   f1=f[idx,ch_1_curr,:]
                   f2=f[idx,ch_2_curr,:]
   ##                f3=f[idx,ch_3_curr,:]
                   f4=f[idx,ch_4_curr,:]
               elif(ch_4==-1):
                   ch_4_curr=-1 #M[ch_4]
                   f4=0 #f[idx,ch_4_curr,:]

                   ch_1_curr=M[ch_1]# find current index
                   ch_2_curr=M[ch_2]
                   ch_3_curr=M[ch_3]
   ##                ch_4_curr=M[ch_4]
                   f1=f[idx,ch_1_curr,:]
                   f2=f[idx,ch_2_curr,:]
                   f3=f[idx,ch_3_curr,:]
   ##                f4=f[idx,ch_4_curr,:]
            except UnboundLocalError:
               print('len(crosslinker_array[idx].chains)',len(crosslinker_array[idx].chains))
               print('k',k)
               stop
            
##            r=bond_register[ch_1,5]#,r2,r3,r4]= bonds_register[[ch_1,ch_2,ch_3,ch_4],5]
##            [fbond,lam_b]=self.get_bondforce(r)
##            fbkT_1 = -fbond*np.linalg.norm(r)/self.K # only multiply by the magnitude of r so that the directions are still preserved
##            r=bond_register[ch_2,5]
##            [fbond,lam_b]=self.get_bondforce(r)
##            fbkT_2 = -fbond*r/self.K
##            r=bond_register[ch_3,5]
##            [fbond,lam_b]=self.get_bondforce(r)
##            fbkT_3 = -fbond*r/self.K
##            r=bond_register[ch_4,5]
##            [fbond,lam_b]=self.get_bondforce(r)
##            fbkT_4 = -fbond*r/self.K

            fit_param=1                        
            f_12_1=np.linalg.norm(f1+f4)
            f_12_2=np.linalg.norm(f1+f2)
            try:
               register_cl_12_1[idx,:]=[tau1*math.exp(-U11 + f_12_1*fit_param*delta11),ch_1,ch_2,ch_3,ch_4,cl_1,cl_2,cl_3,cl_4] # track bothe the chains and the corresponding connected crosslinker
            except UnboundLocalError:
               print('len(crosslinker_array[idx].chains)',len(crosslinker_array[idx].chains))
               print('k',k)
               stop
            register_cl_12_2[idx,:]=[tau1*math.exp(-U12 + f_12_2*fit_param*delta12),ch_1,ch_2,ch_3,ch_4,cl_1,cl_2,cl_3,cl_4]
         

            f_13_1=np.linalg.norm(f1+f3) # this will be due to the effect of f1 and f3 both- the rates of breaking of both bond 1 and bond 3 are the same
            # the break types are different though since the resulting topology is different
##            register_cl_13_1[idx,:]=[tau2*math.exp(-U2 + f_13_1*fit_param*delta2),ch_1,cl_1]
##            f_13_2=np.linalg.norm(f2+f4)
##            register_cl_13_2[idx,:]=[tau2*math.exp(-U2 + f_13_2*fit_param*delta2),ch_2,cl_2]
##            f_13_3=np.linalg.norm(f1+f3)
##            register_cl_13_3[idx,:]=[tau2*math.exp(-U2 + f_13_3*fit_param*delta2),ch_3,cl_3]
##            f_13_4=np.linalg.norm(f2+f4)
##            register_cl_13_4[idx,:]=[tau2*math.exp(-U2 + f_13_4*fit_param*delta2),ch_4,cl_4]

            
##        broken_cl_bonds_register = np.zeros((n_bonds,3)) # the last 2 columns contain the 
        

        # File to write bond broken stats
        if(index_orig%10==0):
           file2 = open('bondbroken_%d.txt'%(index_orig),'w')
           file2.write('#type, atom1, atom2, length, rate(v), t, t_KMC, vmax, active bonds\n') 
       
        # Write probability values in a file (at every KMC call)
        if(pflag==1):
          prob_file = 'prob_%d.txt' %(index_orig)
          fl1 = open(prob_file,'w')   
 
        for i in range (0, n_bonds):
            i_orig=np.where(M==i)[0][0]
            n_i=chain_array[i_orig].n
            
            r = bonds_register[i,5]
            if(r > 0):
              [fbond, lam_b] = self.get_bondforce(r,n_i)
            else: fbond = 0.0

            fit_param = 1
            fbkT = -fbond*r/self.K
            try:
               ch_idx_orig=np.where(M==i)[0][0] # original chain index
            except IndexError:
               print('i',i)
               print('n_bonds',n_bonds)
               print(len(M))
               print(max(M))
##               print(n_bonds)
               print(M)
##               print('n_bonds_orig_and_new',n_bonds_orig_and_new)
               print('n_bonds',n_bonds)
               print('len(np.where(M!=-1)[0])',len(np.where(M!=-1)[0]))
               stop
##            print(ch_idx_orig)
            try:
               n_i=chain_array[ch_idx_orig].n
            except IndexError:
               print('len(chain_array)',len(chain_array))
               print('ch_idx_orig',ch_idx_orig)
##               print('chain_array[ch_idx_orig]',chain_array[ch_idx_orig])
               print('np.where(M==i)',np.where(M==i))
               print('i',i)
               print('M',M)
               stop
            
            bonds_register[i,6] = tau0*math.exp(-U0 + fbkT*fit_param*delta0)#1-(1-tau0*self.N*math.exp(-U0 + fbkT*fit_param*delta0))**(n_i+1)#tau0*math.exp(-U0 + fbkT*fit_param*delta0) #self.N*(n_i+1)* #n_i+1 such bonds can break
##            stop
            broken_cl_bonds_register[i]=n_i*(tau_broken*math.exp(-U_broken + fbkT*fit_param*delta_broken)) # n_i such bonds can break
##            print(broken_cl_bonds_register)
        
            if(pflag==1): fl1.write('%i %i %i %i %i %6.4f %6.4f\n' %(bonds_register[i,0], 
                               bonds_register[i,1], bonds_register[i,2], bonds_register[i,3], 
                               bonds_register[i,4], bonds_register[i,5], bonds_register[i,6]))
        
##        print(broken_cl_bonds_register)
        if(pflag==1): fl1.close()
     
        active_bonds = np.where(bonds_register[:,0]==1) # contains current indices of all chains(connections) which can break
        n_bonds_init = n_bonds#len(active_bonds[0])

        # mapping from current chain index to original chain index in matrix M

        # dealing with chain deletion
##        cnt=0
        n_bonds_orig_and_new=len(M)
##        for i in range(0,n_bonds_orig_and_new):
##            i_curr=M[i]
##            if(i_curr in active_bonds[0]):
##                M[i]=cnt # renumber the chain
##                cnt=cnt+1
##            else:
##                M[i]=-1
##        print('max_cnt',cnt-1)
##        print('len(M)',len(M))
##        print('n_bonds_orig_and_new',n_bonds_orig_and_new)
##        print('n_bonds',n_bonds)
##        print(len(np.where(M!=-1)[0]))
        # dealing with chain addition

                # to be implemented
            
##        print(bonds_register[active_bonds[0],6])
        vmax_chain = max(bonds_register[active_bonds[0],6]) # n_chains
        vmax_broken_cl_chain=max(broken_cl_bonds_register) # sum(n_i) for all chains i
        vmax_12=max(max(register_cl_12_1[:,0]),max(register_cl_12_2[:,0])) # 2*n_links
##        vmax_13=max(max(register_cl_13_1[:,0]),max(register_cl_13_2[:,0]),max(register_cl_13_3[:,0]),max(register_cl_13_4[:,0])) # 4*n_links
        vmax=max(vmax_chain,vmax_broken_cl_chain,vmax_12)#,vmax_13)
##        stop
        '''
        print('bonds_register[active_bonds[0],6]',bonds_register[active_bonds[0],6])
        print('broken_cl_bonds_register',broken_cl_bonds_register)
        print(broken_cl_bonds_register==bonds_register[active_bonds[0],6])
        print('register_cl_12_1[:,0]',register_cl_12_1[:,0])
        print('register_cl_12_2[:,0]',register_cl_12_2[:,0])
        print('register_cl_13_1[:,0]',register_cl_13_1[:,0])
        print('register_cl_13_2[:,0]',register_cl_13_2[:,0])
        print('register_cl_13_3[:,0]',register_cl_13_3[:,0])
        print('register_cl_13_4[:,0]',register_cl_13_4[:,0])
        print('vmax_chain',vmax_chain)
        print('vmax_broken_cl_chain',vmax_broken_cl_chain)
        print('vmax_12',vmax_12)
        print('vmax_13',vmax_13)
        print('vmax',vmax)
        '''
        
##        stop
        sum_n_i=0 #  sum(n_i) for all chains i- only those which are active (existent) right now
        chain_cumulative_sum_n_i=np.zeros(len(active_bonds[0])) # cumulative sum_ni corresponding to each current chain
        # only current chains are considered in this calculation because this is needed only for breaking bonds, and only the current chains contribute to bond breakage
        # and htis thing is recalculated in the loop every time, so no problem with indexing
        
        # for eg. for chain i: the sum value is sum over(n_1+n_2+....+ n_i)
        # this is to identify which broken crosslinker bonds correspond to which chain
        # here- if (broken_bond_number/chain_cumulative_sum_n_i)==0, then that corresponds to chain_i for the lowest possible i value
        for i in range(0,len(active_bonds[0])):
            
##            sum_n_i=sum_n_i+(chain_array[active_bonds[0][i]].n)
            idx_orig=np.where(M==active_bonds[0][i])[0][0]
            sum_n_i=sum_n_i+(chain_array[idx_orig].n)            
            chain_cumulative_sum_n_i[i]=sum_n_i
##            print('i',i)

        
        if(vmax == 0): vmax = 1e-12  
        # if fbkT = 0, vmax = exp(-56). This number below the machine precison.
        # hence, we assign a small detectable number, vmax = 10^{-12}. 
        # Essentially, it implies that bond breaking rate is very low, or 
        # t = 1/(vmax*nbonds) is very high compare to del_t and hence it will not 
        # enter the KMC bond breaking loop 

        n_bonds=len(self.bonds[:,0]) 
        n_atoms=len(self.atoms[:,0])
        num_active_bonds=len(active_bonds[0])+sum_n_i+2*n_atoms+4*n_atoms
        
        t = 1/(vmax*len(active_bonds[0])) 
##        print('KMC statistics:') 
##        print('Max rate, Active bonds, and t_KMC = %6.4E, %5d, %6.4E'%(vmax, len(active_bonds[0]), t))
        if(t < delta_t):
##           print('##############################################')
##           print(crosslinker_array[4757].chains)
##           print('##############################################')
           t = 0
           while(t < delta_t):
                rnd_num  = random.uniform(0,1)
                active_bonds = np.where(bonds_register[:,0]==1)
                                
##                vmax     = max(bonds_register[active_bonds[0],6])
                vmax_chain = max(bonds_register[active_bonds[0],6]) # n_chains
                vmax_broken_cl_chain=max(broken_cl_bonds_register) # sum(n_i) for all chains i
                vmax_12=max(max(register_cl_12_1[:,0]),max(register_cl_12_2[:,0])) # 2*n_links
##                vmax_13=max(max(register_cl_13_1[:,0]),max(register_cl_13_2[:,0]),max(register_cl_13_3[:,0]),max(register_cl_13_4[:,0])) # 4*n_links
                vmax=max(vmax_chain,vmax_broken_cl_chain,vmax_12)#,vmax_13)
                
                
                
                if(vmax == 0): vmax = 1e-12
                
                t_KMC    = 1/(vmax*num_active_bonds)  # IS THIS CORRECT???- I think this is correct!
                
                
                sum_n_i=0 #  sum(n_i) for all chains i
                chain_cumulative_sum_n_i=np.zeros(len(active_bonds[0])) # cumulative sum_ni corresponding to each chain number
                for i in range(0,len(active_bonds[0])):
                    idx_orig=np.where(M==active_bonds[0][i])[0][0]
                    sum_n_i=sum_n_i+(chain_array[idx_orig].n)
                    chain_cumulative_sum_n_i[i]=sum_n_i


                ##IGNORE this comment-- : TO CHECK: len(chain_array) and len(self.links[:,0]) should be the same!! - CHECK THIS!!!!
            
##                n_bonds=len(self.bonds[:,0]) 
                n_atoms=len(self.atoms[:,0])
                num_active_bonds=len(active_bonds[0])+sum_n_i+2*n_atoms+4*n_atoms

                
                bond_index= random.randint(0, num_active_bonds-1)
##                print('bond index',bond_index)
##                print('len(active_bonds[0])',len(active_bonds[0]))
##                print('sum_n_i',sum_n_i)
##                print('n_atoms',n_atoms)
##                print('int(bond_index/len(active_bonds[0]))',int(bond_index/len(active_bonds[0])))
##                print('int(bond_index/(len(active_bonds[0])+sum_n_i))',int(bond_index/(len(active_bonds[0])+sum_n_i)))
##                print('int(bond_index/(len(active_bonds[0])+sum_n_i+2*n_atoms))',int(bond_index/(len(active_bonds[0])+sum_n_i+2*n_atoms)))
##                print('int(bond_index/(len(active_bonds[0])+sum_n_i+2*n_atoms+4*n_atoms))',int(bond_index/(len(active_bonds[0])+sum_n_i+2*n_atoms+4*n_atoms)))
##                
                
                if(int(bond_index/len(active_bonds[0]))==0):# bond in PEG chain is broken
                    num_considered_cumulative[0]=num_considered_cumulative[0]+1
                    # chain i is broken, where i= index of chain corresponding to bonds[bond_index]
                    broken_bond_number=bond_index
                    
                    active_bonds = np.where(bonds_register[:,0]==1) # contains current indices of all chains(connections) which can break
                    # deal with: bonds_register[active_bonds[0],6]
                    #register=bonds_register[active_bonds[0],6]
                    if(broken_bond_number>len(active_bonds[0])):
                        print('bond_index',bond_index)
                        print('num_active_bonds',num_active_bonds)
                        print('n_bonds',n_bonds)
                        print('len(active_bonds[0])',len(active_bonds[0]))
                        stop
                    pot_bond = active_bonds[0][broken_bond_number]
                    rate=bonds_register[pot_bond,6]
                    
##                    print('##################################################')
##                    print('1. (rate/vmax)',(rate/vmax))
##                    print('##################################################')
                    norm_rates_0=1-(1-rate/vmax)**(n_i+1)##((rate/vmax))
##                    stop
                    if((rate/vmax) > rnd_num):
                        num_broken[0]=num_broken[0]+1
                        num_broken_cumulative[0]=num_broken_cumulative[0]+1
                        
                        '''
                        print('bond in PEG chain is broken')
                        print('############################')
                        print('############################')
                        print('crosslinker_array[277].cls',crosslinker_array[277].cls)
                        print('crosslinker_array[277].chains',crosslinker_array[277].chains)
                        print('############################')
                        print('############################')
                        print('chain_array[800].cl_1',chain_array[800].cl_1)

                        print('chain_array[800].cl_2',chain_array[800].cl_2)
                        print('###################')
                        print('rate',rate)
                        print('rate/vmax',rate/vmax)
                        print('###################')
                        '''
                        # bond is broken
                        # update topology accordingly

                        # single chain connection is broken
                        # no addition of chains
                        
                        ch_idx_curr=pot_bond# current
##                        print('ch_idx_curr',ch_idx_curr)
##                        print('M',M)
                        if(ch_idx_curr not in M):
                            print('########################################')
                            print('continuing in case 1')
                            print('########################################')
                            print('ch_idx_curr',ch_idx_curr)
                            print('max(M)',max(M))
                            print('len(M)',len(M))
                            print('M',M)
                            print('len(active_bonds[0])',len(active_bonds[0]))
                            print('n_bonds',n_bonds)
                            print('broken_bond_number',broken_bond_number)
                            for i in M:
                                print(i)
                            stop
                            continue
                        ch_idx_orig=np.where(M==ch_idx_curr)[0][0]
                        
##                        if(ch_idx_orig==800):
##                           print('PROBLEM-800')
##                           stop
##                        print('ch_idx_orig',ch_idx_orig)
##                        print('bonds_register[ch_idx_curr,0]',bonds_register[ch_idx_curr,0])
                        bonds_register[ch_idx_curr,0] = 0   # Bond is broken!
                        ch=chain_array[ch_idx_orig] # current chain class
##                        M[ch_idx_orig]=-1
##                        n_bonds=len(np.where(M!=-1)[0])
##                        n_bonds=n_bonds-1
                        
                           
                        cl_1=ch.cl_1
                        cl_2=ch.cl_2
##                        if( ch_idx_orig==735):
##                           print('PROBLEM - 664/735 in broken cl of PEG chain')
##                           print('cl_1',cl_1)
##                           print('cl_2',cl_2)
                           
##                        print('cl_1',cl_1,'cl_2',cl_2)
                        ch.cl_1=-1 # disconnect the chain from the crosslinker
                        ch.cl_2=-1
##                        if( ch_idx_orig==735):
##                           print('PROBLEM - 664 in broken cl of PEG chain')
##                           print('cl_1',cl_1)
##                           print('cl_2',cl_2)
##                           
##                           stop
                        # disconnect chains from cl_1 and cl_2

##                        print(crosslinker_array[cl_1].chains)
##                        print(crosslinker_array[cl_2].chains)
##                        print(crosslinker_array[cl_1-1].chains)
##                        print(crosslinker_array[cl_2-1].chains)
##                        print(ch_idx_orig)
##                        print('##############################################')
##                        print(crosslinker_array[4757].chains)
##                        print('##############################################')
                        try:
                           index = np.where(np.array(crosslinker_array[cl_1].chains)==ch_idx_orig)[0][0]
                        except IndexError:
                           print('cl_1',cl_1)
                           print('cl_2',cl_2)
                           print('ch_idx_curr',ch_idx_curr)
                           print('ch_idx_orig',ch_idx_orig)
                           print('crosslinker_array[cl_1].chains',crosslinker_array[cl_1].chains)
                           print('active_bonds[0][broken_bond_number]',active_bonds[0][broken_bond_number])
                           print('active_bonds[0]',active_bonds[0])
                           print('broken_bond_number',broken_bond_number)
                           print('len(active_bonds[0])',len(active_bonds[0]))
                           print('num_active_bonds',num_active_bonds)
                           stop
                        
##                        print(index[0][0])
                        
##                        print('crosslinker_array[cl_1].index',crosslinker_array[cl_1].index)
                        crosslinker_array[cl_1].chains[index]=-1
                        crosslinker_array[cl_1].cls[index]=-1
##                        np.delete(cl_1.chains, index)
                        
                        index = np.where(np.array(crosslinker_array[cl_2].chains)==ch_idx_orig)[0][0]
                        crosslinker_array[cl_2].chains[index]=-1
                        crosslinker_array[cl_2].cls[index]=-1
##                        np.delete(cl_2.chains, index)
                        

                        # disconnect crosslinkers cl_1 and cl_2 from each other
                        
##                        index = np.where(np.array(crosslinker_array[cl_1].cls)==cl_2)[0][0]
##                        index = np.where(np.array(crosslinker_array[cl_1].chains)==ch_idx_orig)[0][0]
##                        crosslinker_array[cl_1].cls[index]=-1
##                        np.delete(cl_1.cls, index)

##                        index = np.where(np.array(crosslinker_array[cl_2].cls)==cl_1)[0][0]
##                        index = np.where(np.array(crosslinker_array[cl_2].chains)==ch_idx_orig)[0][0]
                        crosslinker_array[cl_2].cls[index]=-1
##                        np.delete(cl_2.cls, index)
                        
                        # Local Relaxation -- If the bond-broken created a dangling end system
                        # then make the force on the remaining fourth bond
                        link_1 = bonds_register[ch_idx_curr,3]
                        conn = get_link_bonds(link_1, bonds_register)
                        if(len(conn)==3): 
                            if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                 bonds_register[list(conn)[2],6]=0
        
                        elif(len(conn)==2):
                          if(conn[list(conn)[0]]==0):
                             bonds_register[list(conn)[1],6]=0

                        elif(len(conn)==1):
                          bonds_register[list(conn)[0],6]=0


                        link_2 = bonds_register[pot_bond,4]
                        conn = get_link_bonds(link_2, bonds_register)
                        if(len(conn)==3): 
                          if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                             bonds_register[list(conn)[2],6]=0
        
                        elif(len(conn)==2):
                          if(conn[list(conn)[0]]==0):
                             bonds_register[list(conn)[1],6]=0

                        elif(len(conn)==1):
                          bonds_register[list(conn)[0],6]=0
                       
                        
##                    print('n_bonds after PEG chain break',n_bonds)
                    t = t + t_KMC
##                else:  # if nothing is broken
##                   t = t + t_KMC
                   
                elif(int(bond_index/(len(active_bonds[0])+sum_n_i))==0): # bond corresponding to broken crosslinker in chain is broken
                    print('sum_n_i',sum_n_i)
                    print('bond_index',bond_index)
                    print('len(active_bonds[0])',len(active_bonds[0]))
##                    stop
##                    stop
                    num_considered_cumulative[1]=num_considered_cumulative[1]+1
##                    active_bonds = np.where(bonds_register[:,0]==1)
                    # now identify which chain that bond_index corresponds to
                    broken_bond_number=bond_index-(len(active_bonds[0]))
##                    pot_bond=active_bonds[0][broken_bond_number]
                    for i in range(0,len(active_bonds[0])):
##                        print(chain_cumulative_sum_n_i[i])
##                        stop
                        rate_found=False
                        if(chain_cumulative_sum_n_i[i]!=0): # if it is zero, then don't consider for breaking 
                           if(int(broken_bond_number/chain_cumulative_sum_n_i[i])==0):
                               # the chain corresponding to the broken bons is chain _i- update topology accordingly
                               pot_bond=active_bonds[0][i]
   ##                            print('len(broken_cl_bonds_register)',len(broken_cl_bonds_register))
   ##                            print('n_bonds',n_bonds)
   ##                            print(max(active_bonds[0]))
   ##                            print(len(active_bonds[0]))
   ##                            print(np.shape(bonds_register))
                               register=broken_cl_bonds_register[pot_bond] # current indices
                               ch_idx_curr=pot_bond # current idx- this bond is broken 
                               # deal with: broken_cl_bonds_register
                               rate=register
                               rate_found=True
                               break
                    if(rate_found==False):
                       stop
##                    print('##################################################')
##                    print('2. (rate/vmax)',(rate/vmax))
##                    print('##################################################')
                    norm_rates_1=(rate/vmax)
                    if((rate/vmax) > rnd_num):
                        
                        num_broken[1]=num_broken[1]+1
                        num_broken_cumulative[1]=num_broken_cumulative[1]+1

                        '''
                        print('bond corresponding to broken crosslinker in chain is broken')
                        print('############################')
                        print('############################')
                        print('crosslinker_array[277].cls',crosslinker_array[277].cls)
                        print('crosslinker_array[277].chains',crosslinker_array[277].chains)
                        print('############################')
                        print('############################')
                        print('chain_array[800].cl_1',chain_array[800].cl_1)

                        print('chain_array[800].cl_2',chain_array[800].cl_2)
                        print('###################')
                        print('rate',rate)
                        print('rate/vmax',rate/vmax)
                        print('###################')
                        '''
                        # bond is broken
                        # update topology accordingly


                        if(ch_idx_curr not in M):
                            print('########################################')
                            print('continuing in case 2')
                            print('########################################')
                            continue

                        # single chain connection is broken
                        # no addition of chains

                        
                        ch_idx_orig=np.where(M==ch_idx_curr)[0][0]
##                        if(ch_idx_orig==735):
##                           print('PROBLEM - 664 in broken cl of PEG chain')
##                           stop
##                        if(ch_idx_orig==800):
##                           print('PROBLEM-800')
##                           print('ch_idx_curr',ch_idx_curr)
##                           stop
                        
##                        M[ch_idx_orig]=-1
##                        n_bonds=len(np.where(M!=-1)[0])
##                        ch=chain_array[ch_idx_orig] # current chain class
##                        
##                        cl_1=ch.cl_1
##                        cl_2=ch.cl_2
##                        ch.cl_1=-1 # disconnect the chain from the crosslinker
##                        ch.cl_2=-1

                        cl_1=chain_array[ch_idx_orig].cl_1
                        cl_2=chain_array[ch_idx_orig].cl_2

                        chain_array[ch_idx_orig].cl_1=-1
                        chain_array[ch_idx_orig].cl_2=-1

                        # disconnect chains from cl_1 and cl_2
                        try:
                           index = np.where(np.array(crosslinker_array[cl_1].chains)==ch_idx_orig)[0][0]
                        except IndexError:
                           print('cl_1',cl_1)
                           print('cl_2',cl_2)
                           print('crosslinker_array[cl_1].chains',crosslinker_array[cl_1].chains)
                           print('crosslinker_array[cl_2].chains',crosslinker_array[cl_2].chains)
                           print('ch_idx_orig',ch_idx_orig)
                           print('ch_idx_curr',ch_idx_curr)
                           print(bonds_register[ch_idx_curr,0])
                           print('active_bonds[0]',active_bonds[0])
                           stop
                        crosslinker_array[cl_1].chains[index]=-1
                        crosslinker_array[cl_1].cls[index]=-1
##                        np.delete(cl_1.chains, index)
                        
                        index = np.where(np.array(crosslinker_array[cl_2].chains)==ch_idx_orig)[0][0]
                        crosslinker_array[cl_2].chains[index]=-1
                        crosslinker_array[cl_2].cls[index]=-1
##                        np.delete(cl_2.chains, index)
                        bonds_register[ch_idx_curr,0] = 0   # Bond is broken!

                        # disconnect crosslinkers cl_1 and cl_2 from each other
                        
##                        index = np.where(np.array(crosslinker_array[cl_1].cls)==cl_2)[0][0]
##                        index = np.where(np.array(crosslinker_array[cl_1].chains)==ch_idx_orig)[0][0]
##                        crosslinker_array[cl_1].cls[index]=-1
##                        np.delete(cl_1.cls, index)

##                        index = np.where(np.array(crosslinker_array[cl_2].cls)==cl_1)[0][0]
##                        index = np.where(np.array(crosslinker_array[cl_2].chains)==ch_idx_orig)[0][0]
##                        crosslinker_array[cl_2].cls[index]=-1
##                        np.delete(cl_2.cls, index)




                        # Local Relaxation -- If the bond-broken created a dangling end system
                        # then make the force on the remaining fourth bond
                        link_1 = bonds_register[ch_idx_curr,3]
                        conn = get_link_bonds(link_1, bonds_register)
                        if(len(conn)==3): 
                            if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                 bonds_register[list(conn)[2],6]=0
        
                        elif(len(conn)==2):
                          if(conn[list(conn)[0]]==0):
                             bonds_register[list(conn)[1],6]=0

                        elif(len(conn)==1):
                          bonds_register[list(conn)[0],6]=0


                        link_2 = bonds_register[pot_bond,4]
                        conn = get_link_bonds(link_2, bonds_register)
                        if(len(conn)==3): 
                          if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                             bonds_register[list(conn)[2],6]=0
        
                        elif(len(conn)==2):
                          if(conn[list(conn)[0]]==0):
                             bonds_register[list(conn)[1],6]=0

                        elif(len(conn)==1):
                          bonds_register[list(conn)[0],6]=0



                    t = t + t_KMC
##                    print('n_bonds after broken crosslinker in PEG chain is broken',n_bonds)
                    
                elif(int(bond_index/(len(active_bonds[0])+sum_n_i+2*n_atoms))==0): # crosslinker bond is broken based on 1,2 regiochemistry
                    
                    n_i_new=0
                    # addition of 2 new chains
                    # deletion of the 4 original chains connected to the crosslinker

                    num_considered_cumulative[2]=num_considered_cumulative[2]+1
                    
                    broken_bond_number=bond_index-(len(active_bonds[0])+sum_n_i)
                    # identify the crosslinker index which i broken
                    cl_num=int(broken_bond_number/2) # this is the crosslinker which is broken
##                    if(cl_num==390):
##                       
##                       stop

                    if(crosslinker_array[cl_num].cls==[]): # in case the crosslinker is already broken or disconnected from all bonds
                        continue

                   
                    break_type=broken_bond_number%2 #type of break in 12 regiochemistry # if =0, then type_1 and if =1- then type_2

                    
                    # deal with: register_cl_12_1,register_cl_12_2
                    if(break_type==0):
                         register=register_cl_12_1[cl_num]
                    if(break_type==1):
                         register=register_cl_12_2[cl_num]
                         
                    rate=register[0]
                    norm_rates_2=(rate/vmax)
                    if((rate/vmax) > rnd_num):
                        print('sum_n_i before 1,2 regio',sum_n_i)
                        sum_n_i_orig=sum_n_i
                        num_broken[2]=num_broken[2]+1
                        num_broken_cumulative[2]=num_broken_cumulative[2]+1

                        '''
                        print('crosslinker_array[cl_num].cls',crosslinker_array[cl_num].cls)
                        print('crosslinker_array[cl_num].chains',crosslinker_array[cl_num].chains)

                        print('1,2 regiochemistry- chain is broken')
                        print('############################')
                        print('############################')
                        print('crosslinker_array[277].cls',crosslinker_array[277].cls)
                        print('crosslinker_array[277].chains',crosslinker_array[277].chains)
                        print('############################')
##                        print('############################')
##                        print('chain_array[800].cl_1',chain_array[800].cl_1)
##
##                        print('chain_array[800].cl_2',chain_array[800].cl_2)
##                        print('###################')
                        print('rate',rate)
                        print('rate/vmax',rate/vmax)
                        print('###################')
                        '''
                    
                        # bond is broken
                        # update topology accordingly
                        # update the topology accordingly

##                        if(cl_num==141):
##                           print(crosslinker_array[cl_num].cls)
##                           print(crosslinker_array[cl_num].chains)
##                           stop
                           
                        crosslinker_array[cl_num].cls=[] # this crosslinker gets disconnected from all chains and crosslinkers
                        crosslinker_array[cl_num].chains=[]
##                        print('cl_num',cl_num)
                        
                        
                        
                        # 4 chain connections are broken
                        # 2 chains are added
                        

                        # the connecting chains- get the current indices
                        
                        #IMPLEMENT THE CONDITION WHERE SOME OF THESE CHAINS WILL NOT BE PRESENT BECAUSE THE NODE IS NOT FULLY REACTED
##                        print(M)
                        register=np.array([register[0],int(register[1]),int(register[2]),int(register[3]),int(register[4]),int(register[5]),int(register[6]),int(register[7]),int(register[8])],dtype='int')
                        # now convert this array to int to perform all the calculations on the indices
##                        print('register',register)
##                        ch_1_curr=M[register[1]]
                        ch_1_orig=register[1]
##                        ch_2_curr=M[register[2]]
                        ch_2_orig=register[2]
##                        ch_3_curr=M[register[3]]
                        ch_3_orig=register[3]
##                        ch_4_curr=M[register[4]]
                        ch_4_orig=register[4]


                        # the connecting crosslinkers

                        #IMPLEMENT THE CONDITION WHERE SOME OF THESE CROSSLINKERS WILL NOT BE PRESENT BECAUSE THE NODE IS NOT FULLY REACTED
                        cl_1=register[5]
                        cl_2=register[6]
                        cl_3=register[7]
                        cl_4=register[8]
                        
                        
                        
                        if(ch_1_orig==-1):
                           ch_1_curr=-1
                           cl_1=-1
##                           ch_1_orig=register[1]
                        else:
                           ch_1_curr=M[ch_1_orig]
                           if(ch_1_curr not in active_bonds[0]):
                              ch_1_curr=-1
                              cl_1=-1
                        if(ch_2_orig==-1):
                           ch_2_curr=-1
                           cl_2=-1
                           
##                           ch_2_orig=register[2]
                        else:
                           ch_2_curr=M[ch_2_orig]
                           if(ch_2_curr not in active_bonds[0]):
                              ch_2_curr=-1
                              cl_2=-1
                        if(ch_3_orig==-1):
                           ch_3_curr=-1
                           cl_3=-1
##                           ch_3_orig=register[3]
                        else:
                           ch_3_curr=M[ch_3_orig]
                           if(ch_3_curr not in active_bonds[0]):
                              ch_3_curr=-1
                              cl_3=-1
                        if(ch_4_orig==-1):
                           ch_4_curr=-1
                           cl_4=-1
##                           ch_4_orig=register[4]
                        else:
                           ch_4_curr=M[ch_4_orig]
                           if(ch_4_curr not in active_bonds[0]):
                              ch_4_curr=-1
                              cl_4=-1

                        '''
                        print('cl_1',cl_1)
                        print('cl_2',cl_2)
                        print('cl_3',cl_3)
                        print('cl_4',cl_4)
                        print('crosslinker_array[cl_1].chains',crosslinker_array[cl_1].chains)
                        print('crosslinker_array[cl_2].chains',crosslinker_array[cl_2].chains)
                        print('crosslinker_array[cl_3].chains',crosslinker_array[cl_3].chains)
                        print('crosslinker_array[cl_4].chains',crosslinker_array[cl_4].chains)
                        '''


##                        if(ch_1_orig!=-1 and ch_2_orig!=-1 and ch_3_orig!=-1 and ch_4_orig!=-1):
##                        
##                           ch_1_curr=M[ch_1_orig]
####                           ch_1_orig=register[1]
##                           ch_2_curr=M[ch_2_orig]
####                           ch_2_orig=register[2]
##                           ch_3_curr=M[ch_3_orig]
####                           ch_3_orig=register[3]
##                           ch_4_curr=M[ch_4_orig]
##                        if(ch_1_orig==800 or ch_2_orig==800 or ch_3_orig==800 or ch_4_orig==800):
##                           print('PROBLEM-800')
##                           print('break_type',break_type)
##                           print('ch_1_orig',ch_1_orig)
##                           print('ch_2_orig',ch_2_orig)
##                           print('ch_3_orig',ch_3_orig)
##                           print('ch_4_orig',ch_4_orig)
##                           print('ch_1_curr',ch_1_curr)
##                           print('ch_2_curr',ch_2_curr)
##                           print('ch_3_curr',ch_3_curr)
##                           print('ch_4_curr',ch_4_curr)
                           
##                           stop

                        '''
                        print('ch_1_orig',ch_1_orig)
                        print('ch_2_orig',ch_2_orig)
                        print('ch_3_orig',ch_3_orig)
                        print('ch_4_orig',ch_4_orig)
                        print('ch_1_curr',ch_1_curr)
                        print('ch_2_curr',ch_2_curr)
                        print('ch_3_curr',ch_3_curr)
                        print('ch_4_curr',ch_4_curr)
                        '''
                        
##                        if( ch_1_orig==735):
##                           print('PROBLEM - 664 in 1,3')
##                           stop
##                        if( ch_2_orig==735):
##                           print('PROBLEM - 664 in 1,3')
##                           stop
##                        if( ch_3_orig==735):
##                           print('PROBLEM - 664 in 1,3')
##                           stop
##                        if( ch_4_orig==735):
##                           print('PROBLEM - 664 in 1,3')
##                           stop
                        
##                           ch_4_orig=register[4]
##                        if(ch_1_orig!=-1):
##                           if(chain_array[ch_1_orig].cl_1==cl_num):
##                              chain_array[ch_1_orig].cl_1=-1
##                           elif(chain_array[ch_1_orig].cl_2==cl_num):
##                              chain_array[ch_1_orig].cl_2=-1
##                           else:
##                              print('PROBLEM!!')
##                              print('cl_num',cl_num)
##                              print('chain_array[ch_1_orig].cl_1',chain_array[ch_1_orig].cl_1)
##                              print('chain_array[ch_1_orig].cl_2',chain_array[ch_1_orig].cl_2)
##                              print('ch_1_orig',ch_1_orig)
##                              print('ch_1_curr',ch_1_curr)
##                              stop
##                        if(ch_2_orig!=-1):
##                           if(chain_array[ch_2_orig].cl_1==cl_num):
##                              chain_array[ch_2_orig].cl_1=-1
##                           elif(chain_array[ch_2_orig].cl_2==cl_num):
##                              chain_array[ch_2_orig].cl_2=-1
##                           else:
##                              print('PROBLEM!!')
##                              stop
##                        if(ch_3_orig!=-1):
##                           if(chain_array[ch_3_orig].cl_1==cl_num):
##                              chain_array[ch_3_orig].cl_1=-1
##                           elif(chain_array[ch_3_orig].cl_2==cl_num):
##                              chain_array[ch_3_orig].cl_2=-1
##                           else:
##                              print('PROBLEM!!')
##                              stop
##                        if(ch_4_orig!=-1):
##                           if(chain_array[ch_4_orig].cl_1==cl_num):
##                              chain_array[ch_4_orig].cl_1=-1
##                           elif(chain_array[ch_4_orig].cl_2==cl_num):
##                              chain_array[ch_4_orig].cl_2=-1
##                           else:
##                              print('PROBLEM!!')
##                              stop
                           

##                        idx=np.where(chain_array[ch_1_orig].chains==cl_num)[0][0]
##                        chain_array[ch_1_orig].chains[idx]=-1

                        if(ch_1_curr!=-1 and ch_1_curr in active_bonds[0]):
                           bonds_register[ch_1_curr,0] = 0   # Bond is removed
                           # these code snippets are just for debugging. for the code without debugging included, see below this snippet
                           # do this for all types!!
                           if(chain_array[ch_1_orig].cl_1==cl_num):
                              chain_array[ch_1_orig].cl_1=-1
                              chain_array[ch_1_orig].cl_2=-1
                           elif(chain_array[ch_1_orig].cl_2==cl_num):
                              chain_array[ch_1_orig].cl_1=-1
                              chain_array[ch_1_orig].cl_2=-1
                           else:
                              print('PROBLEM!!')
                              print('cl_num',cl_num)
                              print('chain_array[ch_1_orig].cl_1',chain_array[ch_1_orig].cl_1)
                              print('chain_array[ch_1_orig].cl_2',chain_array[ch_1_orig].cl_2)
                              print('ch_1_orig',ch_1_orig)
                              print('ch_1_curr',ch_1_curr)
                              stop

                              
##                           M[ch_1_orig]=-1
##                           n_bonds=len(np.where(M!=-1)[0])

                           # Local Relaxation -- If the bond-broken created a dangling end system
                           # then make the force on the remaining fourth bond
                           link_1 = bonds_register[ch_1_curr,3]
                           conn = get_link_bonds(link_1, bonds_register)
                           if(len(conn)==3): 
                               if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                    bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0


                           link_2 = bonds_register[ch_1_curr,4]
                           conn = get_link_bonds(link_2, bonds_register)
                           if(len(conn)==3): 
                             if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0


                        if(ch_2_curr!=-1 and ch_2_curr in active_bonds[0]):
                           bonds_register[ch_2_curr,0] = 0

                           if(chain_array[ch_2_orig].cl_1==cl_num):
                              chain_array[ch_2_orig].cl_1=-1
                              chain_array[ch_2_orig].cl_2=-1
                           elif(chain_array[ch_2_orig].cl_2==cl_num):
                              chain_array[ch_2_orig].cl_1=-1
                              chain_array[ch_2_orig].cl_2=-1
                           else:
                              print('PROBLEM!!')
                              print('cl_num',cl_num)
                              print('chain_array[ch_2_orig].cl_1',chain_array[ch_2_orig].cl_1)
                              print('chain_array[ch_2_orig].cl_2',chain_array[ch_2_orig].cl_2)
                              print('ch_2_orig',ch_2_orig)
                              print('ch_2_curr',ch_2_curr)
                              stop
                              
##                           M[ch_2_orig]=-1
##                           n_bonds=len(np.where(M!=-1)[0])
                           
                           # Local Relaxation -- If the bond-broken created a dangling end system
                           # then make the force on the remaining fourth bond
                           link_1 = bonds_register[ch_2_curr,3]
                           conn = get_link_bonds(link_1, bonds_register)
                           if(len(conn)==3): 
                               if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                    bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0


                           link_2 = bonds_register[ch_2_curr,4]
                           conn = get_link_bonds(link_2, bonds_register)
                           if(len(conn)==3): 
                             if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0
                             
                        if(ch_3_curr!=-1 and ch_3_curr in active_bonds[0]):   
                           bonds_register[ch_3_curr,0] = 0
                           if(chain_array[ch_3_orig].cl_1==cl_num):
                              chain_array[ch_3_orig].cl_1=-1
                              chain_array[ch_3_orig].cl_2=-1
                           elif(chain_array[ch_3_orig].cl_2==cl_num):
                              chain_array[ch_3_orig].cl_1=-1
                              chain_array[ch_3_orig].cl_2=-1
                           else:
                              print('PROBLEM!!')
                              stop
                              
##                           M[ch_3_orig]=-1
##                           n_bonds=len(np.where(M!=-1)[0])

                           # Local Relaxation -- If the bond-broken created a dangling end system
                           # then make the force on the remaining fourth bond
                           link_1 = bonds_register[ch_3_curr,3]
                           conn = get_link_bonds(link_1, bonds_register)
                           if(len(conn)==3): 
                               if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                    bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0


                           link_2 = bonds_register[ch_3_curr,4]
                           conn = get_link_bonds(link_2, bonds_register)
                           if(len(conn)==3): 
                             if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0



                        if(ch_4_curr!=-1 and ch_4_curr in active_bonds[0]):
                           bonds_register[ch_4_curr,0] = 0
                           if(chain_array[ch_4_orig].cl_1==cl_num):
                              chain_array[ch_4_orig].cl_1=-1
                              chain_array[ch_4_orig].cl_2=-1
                           elif(chain_array[ch_4_orig].cl_2==cl_num):
                              chain_array[ch_4_orig].cl_2=-1
                              chain_array[ch_4_orig].cl_2=-1
                           else:
                              print('PROBLEM!!')
                              print('chain_array[ch_4_orig].cl_1',chain_array[ch_4_orig].cl_1)
                              print('chain_array[ch_4_orig].cl_2',chain_array[ch_4_orig].cl_2)
                              print('cl_num',cl_num)
                              print('ch_4_orig',ch_4_orig)
                              print('ch_4_curr',ch_4_curr)
                              stop
                              
##                           M[ch_4_orig]=-1
##                           n_bonds=len(np.where(M!=-1)[0])

                           # Local Relaxation -- If the bond-broken created a dangling end system
                           # then make the force on the remaining fourth bond
                           link_1 = bonds_register[ch_4_curr,3]
                           conn = get_link_bonds(link_1, bonds_register)
                           if(len(conn)==3): 
                               if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                    bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                               print(list(conn)[0])
                               print(np.shape(bonds_register))
                               bonds_register[list(conn)[0],6]=0


                           link_2 = bonds_register[ch_4_curr,4]
                           conn = get_link_bonds(link_2, bonds_register)
                           if(len(conn)==3): 
                             if(conn[list(conn)[0]]==0 and conn[list(conn)[1]]==0):
                                bonds_register[list(conn)[2],6]=0
           
                           elif(len(conn)==2):
                             if(conn[list(conn)[0]]==0):
                                bonds_register[list(conn)[1],6]=0

                           elif(len(conn)==1):
                             bonds_register[list(conn)[0],6]=0


                             

                        


                        # Newly formed chains:
                        #if break type is 0- chain1- between 1,4, chain2- between 2,3
##                        print('break_type',break_type)
                        
                        if(break_type==0):
                            if(cl_4==-1 and cl_1 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_1].chains)==ch_1_orig)[0][0]
                              crosslinker_array[cl_1].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_1].cls)==cl_num)[0][0]
                              crosslinker_array[cl_1].cls[index]=-1 # disconnected from cl_num
                            if(cl_1==-1 and cl_4 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_4].chains)==ch_4_orig)[0][0]
                              crosslinker_array[cl_4].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_4].cls)==cl_num)[0][0]
                              crosslinker_array[cl_4].cls[index]=-1 # disconnected from cl_num
                              
                            if(cl_3==-1 and cl_2 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_2].chains)==ch_2_orig)[0][0]
                              crosslinker_array[cl_2].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_2].cls)==cl_num)[0][0]
                              crosslinker_array[cl_2].cls[index]=-1 # disconnected from cl_num
                            if(cl_2==-1 and cl_3 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_3].chains)==ch_3_orig)[0][0]
                              crosslinker_array[cl_3].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_3].cls)==cl_num)[0][0]
                              crosslinker_array[cl_3].cls[index]=-1 # disconnected from cl_num
                              
                           ## ADD THE NEW CHAIN ONLY IF PRIMARY LOOP IS NOT GETTING FORMED
                           # OTHERWISE, JUST LEAVE IT- DON'T ADD ANYTHING- BECAUSE WE DON;T WANT PRIMARY LOOPS TO BE THERE IN THE LAMMPS FILE NETWORK BECAUSE IT DOES NOT CONTRIBUTE TO THE ELASTICITY
                           
##                            chains_added=0
                            if(ch_1_orig!=-1 and ch_4_orig!=-1 and (cl_1!=cl_4) and cl_1!=-1 and cl_4!=-1):  # if cl_1==cl_4- that would mean that a secondary loop was involved in this 1,2 regiochemistry, and it is now forming a
                               # if any of the chains is non existent, then no new chain is being formed
                               if(cl_1==-1 or cl_4==-1):
                                    print('cl_1',cl_1)
                                    print('cl_2',cl_2)
                                    print('cl_3',cl_3)
                                    print('cl_4',cl_4)
                                    stop
                               #primary loop, which we don't want to include in the network as it is not elastically effective
##                               print('case_1')
                               ch_new_1_idx= n_bonds#+chains_added# current
##                               chains_added=chains_added+1
                               ch_new_1_idx_orig=len(chain_array)
                               n_i_new=chain_array[register[1]].n+chain_array[register[4]].n+1
                               bonds_register=np.vstack((bonds_register,[1,ch_new_1_idx,n_i_new,cl_1+1,cl_4+1,0,0]))
                               
                               broken_cl_bonds_register=np.append(broken_cl_bonds_register,0)
                               # it is not possible to append the newly created bonds to the active bonds array directly,
                               #and hence- active bonds hve to be calculated each time before a chain is chosen for breaking
                               #in case 1 and 2- might make the code slightly slower, but cant do anything
                               chain_array=np.append(chain_array,chain(ch_new_1_idx_orig,cl_1,cl_4,n_i_new))
                               print('UPDATED chain_array')


                               M=np.append(M,ch_new_1_idx) # add the new chain index current to M- here n_bonds is not getting updated!
##                               print('len(M)',len(M))
##                               print('n_bonds',n_bonds)
##                               print('chain added=',ch_new_1_idx)
                               n_bonds=len(np.where(M!=-1)[0])
##                               print('n_bonds',n_bonds)
                               # reconnect the crosslinkers with chains- update crosslinker_array

                               # ch_1_orig_---> replaced by ch_new_1_idx_orig (for cl_1)
                               # ch_4_orig---> replaced by ch_new_1_idx_orig (for cl_4)
                               if(ch_1_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_1].chains)==ch_1_orig)[0][0] 
                                  crosslinker_array[cl_1].chains[idx]=ch_new_1_idx_orig #  replace one of the earlier chains with the new one, and assign -1 to the other earlier chain
                                  register_cl_12_1[cl_1][idx+1]=ch_new_1_idx_orig # update chain
                                  register_cl_12_2[cl_1][idx+1]=ch_new_1_idx_orig
                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  if(idx==0):
########                                     register_cl_13_1[cl_1][1]=ch_new_1_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_1][1]=ch_new_1_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_1][1]=ch_new_1_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_1][1]=ch_new_1_idx_orig

                                  
                                  # reconnect the crosslinkers with crosslinkers- update crosslinker_array
                                  idx1=np.where(np.array(crosslinker_array[cl_1].cls)==cl_num)[0][0] # DONT USE THIS INDEX 
                                  crosslinker_array[cl_1].cls[idx]=cl_4 #  cl_1 gets connected to cl_4
                                  register_cl_12_1[cl_1][idx+5]=cl_4 # update crosslinker
                                  register_cl_12_2[cl_1][idx+5]=cl_4
                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  if(idx==0):
########                                     register_cl_13_1[cl_1][2]=cl_4
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_1][2]=cl_4
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_1][2]=cl_4
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_1][2]=cl_4

                                  
                                  
##                               print('ch_1_orig',ch_1_orig)
##                               print('ch_4_orig',ch_4_orig)
##                               print('register',register)
##                               print(crosslinker_array[cl_4].chains)

                               if(ch_4_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_4].chains)==ch_4_orig)[0][0] 
                                  crosslinker_array[cl_4].chains[idx]=ch_new_1_idx_orig
                                  register_cl_12_1[cl_4][idx+1]=ch_new_1_idx_orig
                                  register_cl_12_2[cl_4][idx+1]=ch_new_1_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  if(idx==0):
########                                     register_cl_13_1[cl_4][1]=ch_new_1_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_4][1]=ch_new_1_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_4][1]=ch_new_1_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_4][1]=ch_new_1_idx_orig
                                  
                                  idx1=np.where(np.array(crosslinker_array[cl_4].cls)==cl_num)[0][0]  # DONT USE THIS INDEX
                                  crosslinker_array[cl_4].cls[idx]=cl_1
                                  register_cl_12_1[cl_4][idx+5]=cl_1
                                  register_cl_12_2[cl_4][idx+5]=cl_1

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_4][2]=cl_1
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_4][2]=cl_1
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_4][2]=cl_1
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_4][2]=cl_1
                                  
                               
                               if(len(M)!=len(chain_array)):
                                   print('len(M)!=chain_array)')
                                   stop

                            if(ch_2_orig!=-1 and ch_3_orig!=-1 and (cl_2!=cl_3) and cl_3!=-1 and cl_2!=-1):
##                               print('case_2')
                               ch_new_2_idx= n_bonds#+chains_added# current
##                               chains_added=chains_added+1
                               if(cl_2==-1 or cl_3==-1):
                                    print('cl_1',cl_1)
                                    print('cl_2',cl_2)
                                    print('cl_3',cl_3)
                                    print('cl_4',cl_4)
                                    stop
                               ch_new_2_idx_orig=len(chain_array) # chain_array has already been updated with new chain
                               n_i_new=chain_array[register[2]].n+chain_array[register[3]].n+1
                               bonds_register=np.vstack((bonds_register,[1,ch_new_2_idx,n_i_new,cl_2+1,cl_3+1,0,0]))
                               broken_cl_bonds_register=np.append(broken_cl_bonds_register,0)
                               chain_array=np.append(chain_array,chain(ch_new_2_idx_orig,cl_2,cl_3,n_i_new))
                               print('UPDATED chain_array')
##                               print('len(np.where(M!=-1)[0])',len(np.where(M!=-1)[0]))
                               M=np.append(M,ch_new_2_idx) # add the new chain index current to M
##                               print('len(M)',len(M))
##                               print('n_bonds',n_bonds)
##                               print('chain added=',ch_new_2_idx)
                               n_bonds=len(np.where(M!=-1)[0])
##                               print('n_bonds',n_bonds)


                                # reconnect the crosslinkers with chains- update crosslinker_array

                               # ch_2_orig_---> replaced by ch_new_2_idx_orig (for cl_2)
                               # ch_3_orig---> replaced by ch_new_2_idx_orig (for cl_3)
                               if(ch_2_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_2].chains)==ch_2_orig)[0][0] 
                                  crosslinker_array[cl_2].chains[idx]=ch_new_2_idx_orig # replace both of the earlier chains with the new one
                                  register_cl_12_1[cl_2][idx+1]=ch_new_2_idx_orig
                                  register_cl_12_2[cl_2][idx+1]=ch_new_2_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_2][1]=ch_new_2_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_2][1]=ch_new_2_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_2][1]=ch_new_2_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_2][1]=ch_new_2_idx_orig
########                                  
                                  idx1=np.where(np.array(crosslinker_array[cl_2].cls)==cl_num)[0][0] # DONT USE THIS INDEX
                                  crosslinker_array[cl_2].cls[idx]=cl_3 #  cl_1 gets connected to cl_4
                                  register_cl_12_1[cl_2][idx+5]=cl_3
                                  register_cl_12_2[cl_2][idx+5]=cl_3

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_2][2]=cl_3
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_2][2]=cl_3
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_2][2]=cl_3
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_2][2]=cl_3
               

                               if(ch_3_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_3].chains)==ch_3_orig)[0][0] 
                                  crosslinker_array[cl_3].chains[idx]=ch_new_2_idx_orig
                                  register_cl_12_1[cl_3][idx+1]=ch_new_2_idx_orig
                                  register_cl_12_2[cl_3][idx+1]=ch_new_2_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_3][1]=ch_new_2_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_3][1]=ch_new_2_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_3][1]=ch_new_2_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_3][1]=ch_new_2_idx_orig
########                                  
########                                  
                                  # reconnect the crosslinkers with crosslinkers- update crosslinker_array
                                                             
                                  idx1=np.where(np.array(crosslinker_array[cl_3].cls)==cl_num)[0][0] # DONT USE THIS INDEX
                                  crosslinker_array[cl_3].cls[idx]=cl_2
                                  register_cl_12_1[cl_3][idx+5]=cl_2
                                  register_cl_12_2[cl_3][idx+5]=cl_2

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES

########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_3][2]=cl_2
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_3][2]=cl_2
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_3][2]=cl_2
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_3][2]=cl_2


                                            
                               if(len(M)!=len(chain_array)):
                                   print('len(M)!=chain_array)')
                                   stop

                        #if break type is 1- chain1- between 1,2, chain2- between 3,4
                        elif(break_type==1):
                            if(cl_2==-1 and cl_1 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_1].chains)==ch_1_orig)[0][0]
                              crosslinker_array[cl_1].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_1].cls)==cl_num)[0][0]
                              crosslinker_array[cl_1].cls[index]=-1 # disconnected from cl_num
                            if(cl_1==-1 and cl_2 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_2].chains)==ch_2_orig)[0][0]
                              crosslinker_array[cl_2].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_2].cls)==cl_num)[0][0]
                              crosslinker_array[cl_2].cls[index]=-1 # disconnected from cl_num

                            if(cl_4==-1 and cl_3 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_3].chains)==ch_3_orig)[0][0]
                              crosslinker_array[cl_3].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_3].cls)==cl_num)[0][0]
                              crosslinker_array[cl_3].cls[index]=-1 # disconnected from cl_num
                              
                            if(cl_3==-1 and cl_4 !=-1): # in this case- no chain rearragement happens, but for the cl which existed, it becomes disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_4].chains)==ch_4_orig)[0][0]
                              crosslinker_array[cl_4].chains[index]=-1 # disconnected from cl_num
                              index=np.where(np.array(crosslinker_array[cl_4].cls)==cl_num)[0][0]
                              crosslinker_array[cl_4].cls[index]=-1 # disconnected from cl_num
                              
                           
                           
##                            print('ch_1_orig',ch_1_orig)
##                            stop  
##                            chains_added=0
                            if(ch_1_orig!=-1 and ch_2_orig!=-1 and (cl_1!=cl_2) and cl_1!=-1 and cl_2!=-1):
                               if(cl_1==-1 or cl_2==-1):
                                    print('cl_1',cl_1)
                                    print('cl_2',cl_2)
                                    print('cl_3',cl_3)
                                    print('cl_4',cl_4)
                                    stop
                               ch_new_1_idx= n_bonds#+chains_added# current
##                               chains_added=chains_added+1
                               ch_new_1_idx_orig=len(chain_array)
                               n_i_new=chain_array[register[1]].n+chain_array[register[2]].n+1
                               bonds_register=np.vstack((bonds_register,[1,ch_new_1_idx,n_i_new,cl_1+1,cl_2+1,0,0]))
                               broken_cl_bonds_register=np.append(broken_cl_bonds_register,0)
                               chain_array=np.append(chain_array,chain(ch_new_1_idx_orig,cl_1,cl_2,n_i_new))
                               print('UPDATED chain_array')
                               M=np.append(M,ch_new_1_idx) # add the new chain index current to M
##                               print('len(M)',len(M))
##                               print('n_bonds',n_bonds)
##                               print('chain added=',ch_new_1_idx)
                               n_bonds=len(np.where(M!=-1)[0])
##                               print('n_bonds',n_bonds)
                               # reconnect the crosslinkers with chains- update crosslinker_array

                               # ch_1_orig_---> replaced by ch_new_1_idx_orig (for cl_1)
                               # ch_2_orig---> replaced by ch_new_1_idx_orig (for cl_2)
                               if(ch_1_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_1].chains)==ch_1_orig)[0][0] 
                                  crosslinker_array[cl_1].chains[idx]=ch_new_1_idx_orig #  replace both of earlier chains with the new one
                                  register_cl_12_1[cl_1][idx+1]=ch_new_1_idx_orig
                                  register_cl_12_2[cl_1][idx+1]=ch_new_1_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  if(idx==0):
########                                     register_cl_13_1[cl_1][1]=ch_new_1_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_1][1]=ch_new_1_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_1][1]=ch_new_1_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_1][1]=ch_new_1_idx_orig

                                  
                                  # reconnect the crosslinkers with crosslinkers- update crosslinker_array
                                  idx1=np.where(np.array(crosslinker_array[cl_1].cls)==cl_num)[0][0]# DONT USE THIS INDEX
                                  crosslinker_array[cl_1].cls[idx]=cl_2 #  cl_1 gets connected to cl_4
                                  register_cl_12_1[cl_1][idx+5]=cl_2
                                  register_cl_12_2[cl_1][idx+5]=cl_2

                                 # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES                                  if(idx==0):
########                                     register_cl_13_1[cl_1][2]=cl_2
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_1][2]=cl_2
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_1][2]=cl_2
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_1][2]=cl_2
                                  
                                  
                               if(ch_2_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_2].chains)==ch_2_orig)[0][0] 
                                  crosslinker_array[cl_2].chains[idx]=ch_new_1_idx_orig
                                  register_cl_12_1[cl_2][idx+1]=ch_new_1_idx_orig
                                  register_cl_12_2[cl_2][idx+1]=ch_new_1_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES

########                                  if(idx==0):
########                                     register_cl_13_1[cl_2][1]=ch_new_1_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_2][1]=ch_new_1_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_2][1]=ch_new_1_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_2][1]=ch_new_1_idx_orig
                                               
                                   
                                                             
                                  idx1=np.where(np.array(crosslinker_array[cl_2].cls)==cl_num)[0][0] # DONT USE THIS INDEX
                                  crosslinker_array[cl_2].cls[idx]=cl_1
                                  register_cl_12_1[cl_2][idx+5]=cl_1
                                  register_cl_12_2[cl_2][idx+5]=cl_1
                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  if(idx==0):                                     
########                                     register_cl_13_1[cl_2][2]=cl_1
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_2][2]=cl_1
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_2][2]=cl_1
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_2][2]=cl_1

                                                                             
                               if(len(M)!=len(chain_array)):
                                   print('len(M)!=chain_array)')
                                   stop
                            if(ch_3_orig!=-1 and ch_4_orig!=-1 and (cl_3!=cl_4) and cl_3!=-1 and cl_4!=-1):
                               if(cl_3==-1 or cl_4==-1):
                                    print('cl_1',cl_1)
                                    print('cl_2',cl_2)
                                    print('cl_3',cl_3)
                                    print('cl_4',cl_4)
                                    stop
                               ch_new_2_idx= n_bonds#+chains_added# current
##                               chains_added=chains_added+1
                               ch_new_2_idx_orig=len(chain_array) # chain_array has already been updated with new chain
                               n_i_new=chain_array[register[3]].n+chain_array[register[4]].n+1
                               bonds_register=np.vstack((bonds_register,[1,ch_new_2_idx,n_i_new+1,cl_3+1,cl_4+1,0,0]))
                               broken_cl_bonds_register=np.append(broken_cl_bonds_register,0)
                               chain_array=np.append(chain_array,chain(ch_new_2_idx_orig,cl_3,cl_4,n_i_new))
                               print('UPDATED chain_array')
##                               print(chain_array[-1].n)
##                               stop
                               M=np.append(M,ch_new_2_idx) # add the new chain index current to M
##                               print('len(M)',len(M))
##                               print('n_bonds',n_bonds)
##                               print('chain added=',ch_new_2_idx)
                               n_bonds=len(np.where(M!=-1)[0])
##                               print('n_bonds',n_bonds)

                               # reconnect the crosslinkers with chains- update crosslinker_array

                               # ch_3_orig_---> replaced by ch_new_2_idx_orig (for cl_3)
                               # ch_4_orig---> replaced by ch_new_2_idx_orig (for cl_4)
                               if(ch_3_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_3].chains)==ch_3_orig)[0][0] 
                                  crosslinker_array[cl_3].chains[idx]=ch_new_2_idx_orig #  replace both of earlier chains with the new one
                                  register_cl_12_1[cl_3][idx+1]=ch_new_2_idx_orig
                                  register_cl_12_2[cl_3][idx+1]=ch_new_2_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_3][1]=ch_new_2_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_3][1]=ch_new_2_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_3][1]=ch_new_2_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_3][1]=ch_new_2_idx_orig
########                                  
########                                  


                                  
                                  # reconnect the crosslinkers with crosslinkers- update crosslinker_array
                                  idx1=np.where(np.array(crosslinker_array[cl_3].cls)==cl_num)[0][0]  # DON'T USE THIS INDEX
                                  crosslinker_array[cl_3].cls[idx]=cl_4 #  cl_1 gets connected to cl_4
                                  register_cl_12_1[cl_3][idx+5]=cl_4
                                  register_cl_12_2[cl_3][idx+5]=cl_4

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_3][2]=cl_4
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_3][2]=cl_4
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_3][2]=cl_4
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_3][2]=cl_4
########                                  

                               if(ch_4_curr!=-1):
                                  idx=np.where(np.array(crosslinker_array[cl_4].chains)==ch_4_orig)[0][0] 
                                  crosslinker_array[cl_4].chains[idx]=ch_new_2_idx_orig
                                  register_cl_12_1[cl_4][idx+1]=ch_new_2_idx_orig
                                  register_cl_12_2[cl_4][idx+1]=ch_new_2_idx_orig

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES
########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_4][1]=ch_new_2_idx_orig
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_4][1]=ch_new_2_idx_orig
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_4][1]=ch_new_2_idx_orig
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_4][1]=ch_new_2_idx_orig
########                                               
                                                             
                                  idx1=np.where(np.array(crosslinker_array[cl_4].cls)==cl_num)[0][0] # DONT USE THIS INDEX 
                                  crosslinker_array[cl_4].cls[idx]=cl_3
                                  register_cl_12_1[cl_4][idx+5]=cl_3
                                  register_cl_12_2[cl_4][idx+5]=cl_3

                                  # THE FOLLOWING LINES HAVE BEEN COMMENTED OUT BECAUSE THERE IS NO 1,3 REGIO IN ETHER CASE
                                  # IN CASE 1,3 REGIO IS INCLUDED IN ETHER, THEN UNCOMMENT THESE LINES

########                                  
########                                  if(idx==0):
########                                     register_cl_13_1[cl_4][2]=cl_3
########                                  elif(idx==1):
########                                     register_cl_13_2[cl_4][2]=cl_3
########                                  elif(idx==2):
########                                     register_cl_13_3[cl_4][2]=cl_3
########                                  elif(idx==3):
########                                     register_cl_13_4[cl_4][2]=cl_3
                                  

                                            
                               if(len(M)!=len(chain_array)):
                                   print('len(M)!=chain_array)')
                                   stop



                        sum_n_i_new=0
                        active_bonds = np.where(bonds_register[:,0]==1)
                        for i in range(0,len(active_bonds[0])):
                     
                  ##            sum_n_i=sum_n_i+(chain_array[active_bonds[0][i]].n)
                          idx_orig=np.where(M==active_bonds[0][i])[0][0]
                          sum_n_i_new=sum_n_i_new+(chain_array[idx_orig].n)
                        print('sum_n_i_updated',sum_n_i_new)
                        print('n_i_added=',sum_n_i_new-sum_n_i_orig)
##                        stop
##                        if(sum_n_i_new-sum_n_i_orig==1):
##                           print([cl_1,cl_2,cl_3,cl_4])
##                           if(len(np.where(np.array([cl_1,cl_2,cl_3,cl_4])==-1)[0])==0):
####                           print(sum_n_i_new-sum_n_i_orig)
##                              
##                              stop
##                        if(sum_n_i_new-sum_n_i_orig<0): # this can happen when there are not
                           #enough chains around the crosslinker, (eg- only one chain or 2 chains, and break type breaks the crosslinker in a way that both these chains separate) and so when
                           #that crosslinker breaks- it creates a dangling end and basically removed the chain
                           #from the active array- so n_i decreases
##                           print([cl_1,cl_2,cl_3,cl_4])
##                           if(len(np.where(np.array([cl_1,cl_2,cl_3,cl_4])==-1)[0])==0):
####                           print(sum_n_i_new-sum_n_i_orig)
                              
##                              stop
                        print('n_i_new',n_i_new)
##                        if(n_i_new==0):
##                           stop
    ####                    #ADD THIS CODE!!
    ####
    ####
    ####                    
    ####                    ## ADD THE CODE FOR UPDATING n_i FOR THE NEWLY FORMED CHAINS
    ####                    
                    
                    t = t + t_KMC
                        
                       

                    
            
##                    print('n_bonds: 1,2 regiochemistry',n_bonds)
##                    if(cl_num==77):
##                       stop

                    
########                elif(int(bond_index/(len(active_bonds[0])+sum_n_i+2*n_atoms+4*n_atoms))==0): # crosslinker bond is broken based on 1,3 regiochemistry
########                     num_broken[3]=num_broken[3]+1
########                     num_considered_cumulative[3]=num_considered_cumulative[3]+1
########                    # bond will be broken
########                    # no bond additions will be there
########                     broken_bond_number=bond_index-(len(active_bonds[0])+sum_n_i+2*n_atoms)
########                     cl_num=int(broken_bond_number/4) #
##########                     print('cl_num',cl_num)
##########                     if(cl_num==390):
##########                       stop
########
########                     if(crosslinker_array[cl_num].cls==[]): # in case the crosslinker is already broken or disconnected from all bonds
########                        continue
########                     
##########                     if(cl_num>5000):
##########                         print('n_atoms',n_atoms)
##########                         print('n_bonds',n_bonds)
##########                         print('broken_bond_number',broken_bond_number)
##########                         print('bond_index',bond_index)
##########                        
##########                         print('sum_n_i',sum_n_i)
##########                         print('num_active_bonds',num_active_bonds)
########                     break_type=broken_bond_number%4 #type of break in 13 regiochemistry # if =0, then type_1,  if =1- then type_2  , if =2- then type_3, if =3- then type_4
##########                     print('break_type',break_type)
########
########                     
########                     # deal with: register_cl_13_1,register_cl_13_2,register_cl_13_3,register_cl_13_4
########                     if(break_type==0):
##########                         print('break type=0')
##########                         print(len(np.where(np.array(crosslinker_array[cl_num].chains)!=-1)))
##########                         print((crosslinker_array[cl_num].chains))
########                         if(len(np.where(np.array(crosslinker_array[cl_num].chains)!=-1)[0])<1):
########                             continue
########                         else:
########                             register=register_cl_13_1[cl_num]
########                             ch_idx_orig=int(register[1])#crosslinker_array[cl_num].chains[0]
##########                             print('ch_idx_orig',ch_idx_orig)
##########                             print('break_type_0')
##########                             print(len(np.where(np.array(crosslinker_array[cl_num].chains)!=-1)[0]))
##########                             print(np.array(crosslinker_array[cl_num].chains))
########                     if(break_type==1):
########                         if(len(np.where(np.array(crosslinker_array[cl_num].chains)!=-1)[0])<2):
########                             continue
########                         else:
########                             register=register_cl_13_2[cl_num]
########                             ch_idx_orig=int(register[1])#crosslinker_array[cl_num].chains[1]
##########                             print('ch_idx_orig',ch_idx_orig)
##########                             print('break_type_1')
########                     if(break_type==2):
########                         if(len(np.where(np.array(crosslinker_array[cl_num].chains)!=-1)[0])<3):
########                             continue
########                         else:
########                             register=register_cl_13_3[cl_num]
########                             ch_idx_orig=int(register[1])#crosslinker_array[cl_num].chains[2]
##########                             print('ch_idx_orig',ch_idx_orig)
##########                             print('break_type_2')
########                     if(break_type==3):
########                         if(len(np.where(np.array(crosslinker_array[cl_num].chains)!=-1)[0])<4):
########                             continue
########                         else:
########                             register=register_cl_13_4[cl_num]
##########                             print('cl_num',cl_num)
##########                             print('len(crosslinker_array)',len(crosslinker_array))
##########                             print('len(crosslinker_array[cl_num].chains)',len(crosslinker_array[cl_num].chains))
########                             ch_idx_orig=int(register[1])#crosslinker_array[cl_num].chains[3]
##########                             print('ch_idx_orig',ch_idx_orig)
##########                             print('break_type_3')
########
########                     rate=register[0]
########                     register=np.array([register[0],int(register[1]),int(register[2])],dtype='int')
########                     
########                     ch_idx_curr=M[ch_idx_orig]# find current index--- ## THIS BOND IS BROKEN!!
########                     
##########                     if(ch_idx_curr not in active_bonds[0]):
##########                        stop
##########                     print('bonds_register[ch_idx_curr,0] ',bonds_register[ch_idx_curr,0] )
########                     norm_rates_3=(rate/vmax)
########                     if((rate/vmax) > rnd_num and ch_idx_orig!=-1 and ch_idx_curr in active_bonds[0]):
##########                        print('bonds_register[ch_idx_curr,0] ',bonds_register[ch_idx_curr,0] )
##########                        print('chain_array[ch_idx_orig].cl_1',chain_array[ch_idx_orig].cl_1)
##########                        print('chain_array[ch_idx_orig].cl_2',chain_array[ch_idx_orig].cl_2)
##########                        print('cl_num',cl_num)
##########                        print('crosslinker_array[cl_num].chains',crosslinker_array[cl_num].chains)
##########                        if(ch_idx_orig==800):
##########                           print('PROBLEM-800')
##########                           stop
########                        num_broken[3]=num_broken[3]+1
########                        num_broken_cumulative[3]=num_broken_cumulative[3]+1
##########                        print('1,3 regiochemistry- chain is broken')
##########                        print('############################')
##########                        print('############################')
##########                        print('crosslinker_array[277].cls',crosslinker_array[277].cls)
##########                        print('crosslinker_array[277].chains',crosslinker_array[277].chains)
##########                        print('############################')
##########                        print('############################')
##########                        print('chain_array[800].cl_1',chain_array[800].cl_1)
##########
##########                        print('chain_array[800].cl_2',chain_array[800].cl_2)
##########                        print('###################')
##########                        print('rate',rate)
##########                        print('rate/vmax',rate/vmax)
##########                        print('###################')
########                    
########                        # bond is broken
########                        # update topology accordingly
########
########                        # single chain connection is broken
########                        # no addition
########
########                        # update the topology accordingly
########                        
##########                        if( ch_idx_orig==735):
##########                           print('PROBLEM - 664 in 1,3')
##########                           stop
########                        bonds_register[ch_idx_curr,0] = 0   # Bond is broken!
########
##########                        if
##########                        M[ch_idx_curr]=-1
##########                        n_bonds=len(np.where(M!=-1)[0])
########
########                        
##########                        ch=chain_array[ch_idx_orig] # current chain class
########                        cl_1=chain_array[ch_idx_orig].cl_1
########                        cl_2=chain_array[ch_idx_orig].cl_2
##########                        cl_1=ch.cl_1
##########                        cl_2=ch.cl_2
########
########                        chain_array[ch_idx_orig].cl_1=-1
########                        chain_array[ch_idx_orig].cl_2=-1
########
##########                        ch.cl_1=-1 # disconnect the chain from the crosslinker
##########                        ch.cl_2=-1
########                        
########
########                        # disconnect crosslinkers cl_1 and cl_2 from each other
########                        # IN CASE THERE IS A SECONDARY LOOP, ie. 2 BONDS EXIST BETWEEN CL_1 AND CL_2, THEN IN THAT CASE,
########                        # BOTH THE CROSSLINKERS WILL NOT BE COMPLETELY DISCONNECTED
########                        # IN THIS CASE- i WILL HAVE TO GET THE INDEX CORRECTLY
##########                        try:
##########                        index = np.where(np.array(crosslinker_array[cl_1].cls)==cl_2)[0]
########                        index = np.where(np.array(crosslinker_array[cl_1].chains)==ch_idx_orig)[0]
########                        
########                        try:
########                           if(crosslinker_array[cl_1].chains[index[0]]==ch_idx_orig):
########                              crosslinker_array[cl_1].cls[index[0]]=-1  # replace that connection by -1, don't change the length of the array, else the infor about position of chain/crosslinker -1,2,3,4- will be lost
########                           elif(crosslinker_array[cl_1].chains[index[1]]==ch_idx_orig):
########                              crosslinker_array[cl_1].cls[index[1]]=-1
########                           else:
########                              stop
########                        except IndexError:
########                           print('crosslinker_array[cl_1].cls',crosslinker_array[cl_1].cls)
########                           print('crosslinker_array[cl_2].cls',crosslinker_array[cl_2].cls)
########                           print('cl_1',cl_1)
########                           print('cl_2',cl_2)
########                           print('cl_num',cl_num)
########                           print('ch_idx_orig',ch_idx_orig)
########                           print('ch_idx_curr',ch_idx_curr)
########                           stop
##########                        cl_1.cls=np.delete(cl_1.cls, index)
########
##########                        index = np.where(np.array(crosslinker_array[cl_2].cls)==cl_1)[0]
########                        index = np.where(np.array(crosslinker_array[cl_2].chains)==ch_idx_orig)[0]
########                        
########                        if(crosslinker_array[cl_2].chains[index[0]]==ch_idx_orig):
########                           crosslinker_array[cl_2].cls[index[0]]=-1  # replace that connection by -1, don't change the length of the array, else the infor about position of chain/crosslinker -1,2,3,4- will be lost
########                        elif(crosslinker_array[cl_2].chains[index[1]]==ch_idx_orig):
########                           crosslinker_array[cl_2].cls[index[1]]=-1
########                        else:
########                           stop
########
########                        
########                        
########                        
########
##########                        chain_array[ch_idx_orig].cl_1=-1
##########                        chain_array[ch_idx_orig].cl_2=-1
########
########                        # disconnect chains from cl_1 and cl_2
##########                        print('ch_idx_orig',ch_idx_orig)
##########                        print('ch_idx_curr',ch_idx_curr)
##########                        print('crosslinker_array[cl_1].chains',crosslinker_array[cl_1].chains)
##########                        print('crosslinker_array[cl_2].chains',crosslinker_array[cl_2].chains)
##########                        print('cl_1',cl_1)
##########                        print('cl_2',cl_2)
########                        index = np.where(np.array(crosslinker_array[cl_1].chains)==ch_idx_orig)[0][0]
########                        crosslinker_array[cl_1].chains[index]=-1
##########                        cl_1.chains=np.delete(cl_1.chains, index)
##########                        print('cl_num',cl_num)
########                        index = np.where(np.array(crosslinker_array[cl_2].chains)==ch_idx_orig)[0][0]
########                        crosslinker_array[cl_2].chains[index]=-1
##########                        cl_2.chains=np.delete(cl_2.chains, index)
########
########
########                       
########                           
##########                        crosslinker_array[cl_2].cls[index]=-1 
##########                        cl_2.cls=np.delete(cl_2.cls, index)
########  
########
########                         
########                     t = t + t_KMC
##                     print('n_bonds 1,3 regiochemistry',n_bonds)

                     

                     # LOCAL RELAXATION TO BE IMPLEMENTED!!!
                    
                else:  # if nothing is broken
                   t = t + t_KMC
                if(index_orig%10==0):
##
##                         # WRITE TO FILE TO BE MODIFIED
##                         print(index)
##                         file2.write('%5d  %5d  %5d  %0.4E  %0.4E  %0.4E  %0.4E  %0.4E  %5d\n'%(bonds_register[pot_bond,2], bonds_register[pot_bond,3], 
##                           bonds_register[pot_bond,4], bonds_register[pot_bond,5], bonds_register[pot_bond,6], 
##                           t, t_KMC, vmax, len(active_bonds[0])) )
                         file2.flush()

                
                active_bonds = np.where(bonds_register[:,0]==1)
                
        sum_n_i_new=0
        active_bonds = np.where(bonds_register[:,0]==1)
        for i in range(0,len(active_bonds[0])):
            
##            sum_n_i=sum_n_i+(chain_array[active_bonds[0][i]].n)
            idx_orig=np.where(M==active_bonds[0][i])[0][0]
            sum_n_i_new=sum_n_i_new+(chain_array[idx_orig].n)            
##            chain_cumulative_sum_n_i[i]=sum_n_i
        
##        print('M before deletion update',M)
        # dealing with chain deletion
        cnt=0
        n_bonds_orig_and_new=len(M)
        for i in range(0,n_bonds_orig_and_new):
            i_curr=M[i]
            if(i_curr in active_bonds[0]):
                M[i]=cnt
                cnt=cnt+1
            else:
                M[i]=-1
##                if(i==800):
##                   stop
##        print('#################################')
##        print('M[800]',M[800])
##        print('#################################')
##        print('M after deletion update',M)
##        print('len(M)',len(M))
##         print('n_bonds_orig_and_new',n_bonds_orig_and_new)
##        print('n_bonds',n_bonds)
##        print(len(np.where(M!=-1)[0]))

            # dealing with chain addition

            # to be implemented

        if(index_orig%10==0): file2.close()
    
        n_bonds_final = len(active_bonds[0])
        if(n_bonds_final < n_bonds_init): # removing bonds from network after break- modify topology
           bonds_final = np.zeros((n_bonds_final, 4), dtype = int)
           bonds_final[:,0:4] = bonds_register[active_bonds[0],1:5].astype(int)
           self.bonds = bonds_final
 
####        print('time, init bonds, final bonds = %6.4E, %5d, %5d'%(t, n_bonds_init, n_bonds_final))
####        print('---------------------------------------------------------------')
####        print('len(M)',len(M))
######         print('n_bonds_orig_and_new',n_bonds_orig_and_new)
####        print('len(self.bonds)',len(self.bonds))
####        print('len(np.where(M!=-1)[0])',len(np.where(M!=-1)[0]))
        if(len(np.where(M!=-1)[0])!=len(self.bonds)):
           print('len(np.where(M!=-1)[0])!=len(self.bonds)')
           print('n_bonds',n_bonds)
           stop
####        print('sum_n_i',sum_n_i)
####        print('sum_n_i_new',sum_n_i_new)
        
####        print('num_broken',num_broken)
        return t, n_bonds_init, n_bonds_final, M, chain_array, crosslinker_array, num_broken, num_broken_cumulative,num_considered_cumulative, sum_n_i,[norm_rates_0,norm_rates_1,norm_rates_2,norm_rates_3]
 
