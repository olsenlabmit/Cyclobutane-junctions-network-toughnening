#!/use/local/bin/env python
# -*- coding: utf-8 -*-

# Python script to write configuration for LAMMPS

import math
import numpy as np

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

        

def readLAMMPS(filename, N, vflag):

   f1=open(filename,"r")

   line1 = f1.readline()
   line2 = f1.readline()

   line3 = f1.readline()
   line3 = line3.strip()
   n_links = int(line3.split(" ")[0])
 
   line4 = f1.readline()
   line4 = line4.strip()
   atom_types = int(line4.split(" ")[0])

   line5 = f1.readline()
   line5 = line5.strip()
   n_chains = int(line5.split(" ")[0])

   # The following two arrays will contain the objects of classes crosslinker and chain respectively
   crosslinker_array=[]#np.zeros(n_links)
   chain_array=[]#np.zeros(n_chains)

   line6 = f1.readline()
   line6 = line6.strip()
   bond_types = int(line6.split(" ")[0])

   links_unsort  = np.zeros((n_links,4))
   links   = np.zeros((n_links,3), dtype = float)
   chains  = np.full((n_chains,4), -1, dtype = int)
   mass    = np.zeros(atom_types, dtype = float)

   line7 = f1.readline()
   line8 = f1.readline()
   line8 = line8.strip()
   xlo = float(line8.split(" ")[0])
   xhi = float(line8.split(" ")[1])

   line9 = f1.readline()
   line9 = line9.strip()
   ylo = float(line9.split(" ")[0])
   yhi = float(line9.split(" ")[1])

   line10 = f1.readline()
   line10 = line10.strip()
   zlo = float(line10.split(" ")[0])
   zhi = float(line10.split(" ")[1])


   for i in range (0, 3):
       f1.readline()
   
   for i in range(0, atom_types):
       line = f1.readline()
       line = line.strip()
       mass[i] = float(line.split(" ")[1])

   f1.close()


   links_unsort = np.genfromtxt(filename, usecols=(0,3,4,5), skip_header=18, max_rows=n_links)

   for i in range(0, n_links):
       index = int(links_unsort[i,0])
       links[index-1,:] = links_unsort[i,1:4]
       # update the crosslinker_array when a new crosslinker is detected
       crosslinker_array.append(crosslinker(index-1))
   

   chains[:,0] = N

   if(vflag==0):
      chains[:,1:4] = np.genfromtxt(filename, usecols=(1,2,3), skip_header=17+n_links+3, max_rows=n_chains)
   elif(vflag==1):
      chains[:,1:4] = np.genfromtxt(filename, usecols=(1,2,3), skip_header=17+2*n_links+2*3, max_rows=n_chains)
   else:
      print("Invalid Velocity Flag")

   chain_idx=0
   for i in chains:
      # update the chain_array when a new chain is detected
      
      chain_array.append(chain(chain_idx,i[2]-1,i[3]-1)) # add the indices, not the actual values (there is a difference of 1)
##      if(i[2]==i[3]):
##         print('i[2]-1',i[2]-1)
##         print('i[3]-1',i[3]-1)
      # update the connections for the two corresponding crosslinkers
      cl_1=crosslinker_array[i[2]-1] # cl_1 of chain
      cl_2=crosslinker_array[i[3]-1] # cl_2 of chain

      
      # update for crosslinker 1 - i[2]
      cl_1.chains.append(chain_idx)
##      if(cl_2.index in cl_1.cls):
##         print('cl_1.index',cl_1.index)
##         print('cl_1.cls',cl_1.cls)
##         print('cl_2.index',cl_2.index)
##         stop
      cl_1.cls.append(cl_2.index)
##      # update for crosslinker 2 - i[3]
      cl_2.chains.append(chain_idx)
##      if(cl_1.index in cl_2.cls):
##         print('cl_2.index',cl_2.index)
##         print('cl_2.cls',cl_2.cls)
##         print('cl_1.index',cl_1.index)
##         stop
      cl_2.cls.append(cl_1.index)
      
      chain_idx=chain_idx+1

   loop_atoms = np.genfromtxt('primary_loops', usecols=(1), skip_header=0)
   loop_atoms.tolist()
##   stop
##   print('crosslinker_array[390].chains',crosslinker_array[390].chains)
##   print('crosslinker_array[390].cls',crosslinker_array[390].cls)

   return xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, links, chains, atom_types, bond_types, mass, loop_atoms, crosslinker_array, chain_array



def writeLAMMPS(filename, xlo, xhi, ylo, yhi, zlo, zhi, links, chains, atom_types, bond_types, mass, loop_atoms):
   
   n_atoms  = len(links[:,0])
   n_bonds  = len(chains[:,0])
   n_loops  = len(loop_atoms)

   # LAMMPS input file begins here
   file1=open(filename,"w")
   file1.write("# Gels-Network -- Initial Configuration\n")
   file1.write("\n")
   file1.write("%i %s\n" % (n_atoms,"atoms"))
   file1.write("%i %s\n" % (atom_types,"atom types"))
   file1.write("%i %s\n" % (n_bonds,"bonds"))
   file1.write("%i %s\n" % (bond_types,"bond types"))
   file1.write("\n")
   file1.write("%7.4f %7.4f %s %s\n" % (xlo,xhi,"xlo","xhi"))
   file1.write("%7.4f %7.4f %s %s\n" % (ylo,yhi,"ylo","yhi"))
   file1.write("%7.4f %7.4f %s %s\n" % (zlo,zhi,"zlo","zhi"))
   
   file1.write("\n")
   file1.write("Masses\n")
   file1.write("\n")
   for i in range(0, atom_types):
       file1.write("%i %6.4f\n" % (i+1,mass[i]))
   
   file1.write("\n")
   file1.write("Atoms\n")
   file1.write("\n")
   cnt = 0
   cnt_loops = 0
   for i in range(0, n_atoms):
      # cnt = cnt + 1
      # file1.write("{:2} {:2} {:2} {:7} {:7} {:7}\n".format(cnt,1,1,links[i,0],links[i,1],links[i,2]))
       if(cnt_loops < n_loops and cnt == loop_atoms[cnt_loops]):
          cnt = cnt + 1
          cnt_loops = cnt_loops + 1
          file1.write("{:2} {:2} {:2} {:7} {:7} {:7}\n".format(cnt,1,2,links[i,0],links[i,1],links[i,2]))
       else:
          cnt = cnt + 1
          file1.write("{:2} {:2} {:2} {:7} {:7} {:7}\n".format(cnt,1,1,links[i,0],links[i,1],links[i,2]))
   
   
   # Bond writing starts here
   file1.write("\n")
   file1.write("Bonds\n")
   file1.write("\n")
   cnt = 0
   for i in range(0, n_bonds):
       cnt = cnt + 1
       file1.write("{:4} {:4} {:4} {:4}\n".format(cnt,1,chains[i,2], chains[i,3]))
   
   
   file1.close()



#def writeLAMMPSafternetgen(filename, xlo, xhi, ylo, yhi, zlo, zhi, n_links, n_chains, n_loops, n_dang, n_free, n_break, links, chains, atom_types, bond_types, mass):
def writeLAMMPSafternetgen(filename, xlo, xhi, ylo, yhi, zlo, zhi, links, chains, atom_types, bond_types, mass, loop_atoms):

#   n_atoms  = n_links
#   n_bonds  = n_chains - n_loops - n_dang - n_free -n_break
 
   n_atoms = len(links[:,0])  
   n_bonds = len(chains[:,0])  
   n_loops = len(loop_atoms)

   print(n_atoms, n_bonds, n_loops)
 
   # LAMMPS input file begins here
   file1=open(filename,"w")
   file1.write("# Gels-Network -- Initial Configuration\n")
   file1.write("\n")
   file1.write("%i %s\n" % (n_atoms,"atoms"))
   file1.write("%i %s\n" % (atom_types,"atom types"))
   file1.write("%i %s\n" % (n_bonds,"bonds"))
   file1.write("%i %s\n" % (bond_types,"bond types"))
   file1.write("\n")
   file1.write("%7.4f %7.4f %s %s\n" % (xlo,xhi,"xlo","xhi"))
   file1.write("%7.4f %7.4f %s %s\n" % (ylo,yhi,"ylo","yhi"))
   file1.write("%7.4f %7.4f %s %s\n" % (zlo,zhi,"zlo","zhi"))
   
   file1.write("\n")
   file1.write("Masses\n")
   file1.write("\n")
   for i in range(0, atom_types):
       file1.write("%i %6.4f\n" % (i+1,mass[i]))
   
   file1.write("\n")
   file1.write("Atoms\n")
   file1.write("\n")
   cnt = 0
   cnt_loops = 0
   for i in range(0, n_atoms):
       if(cnt_loops < n_loops and cnt == loop_atoms[cnt_loops]): 
          cnt = cnt + 1
          cnt_loops = cnt_loops + 1
          file1.write("{:2} {:2} {:2} {:7} {:7} {:7}\n".format(cnt,1,2,links[i,0],links[i,1],links[i,2]))
       else:
          cnt = cnt + 1
          file1.write("{:2} {:2} {:2} {:7} {:7} {:7}\n".format(cnt,1,1,links[i,0],links[i,1],links[i,2]))
   
  
   print(cnt, cnt_loops) 
   # Bond writing starts here
   file1.write("\n")
   file1.write("Bonds\n")
   file1.write("\n")
   cnt = 0
#   for i in range(0, n_chains):
#         if(chains[i,2]!=-1):
#            if(chains[i,1] != chains[i,2]):
#               cnt = cnt + 1
#               file1.write("{:4} {:4} {:4} {:4}\n".format(cnt,1,chains[i,1], chains[i,2]))
   
   for i in range(0, n_bonds):
       cnt = cnt + 1
       file1.write("{:4} {:4} {:4} {:4}\n".format(cnt,1,chains[i,1], chains[i,2]))
   
   file1.close()
