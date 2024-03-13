N=33.6
b=1
func=4

n_chains  = 10000
cR3=4.37 # dimless conc

conc=cR3/(N*b**2)**1.5 #(chains/nm3)
L=(n_chains/conc)**(1/3)
C_mM=conc*1000/0.6022 # conc in mM
##L = 32.5984
##print(L)

factor=1



U_PEG=100*factor
tau_PEG=1
delta_PEG=0.1


    
junction='ctrl'

if(junction=='es'):
   
    # PEG chain 
    U0=U_PEG
    tau0=tau_PEG
    delta0=delta_PEG

    # CB-junction 1,2 regiochemistry- ether and ester
    U11=1*factor
    U12=U11
    tau1=1
    delta1=0.1

    # CB-junction 1,3 regiochemistry- ester and control
    U2=1.0*factor
    tau2=1
    delta2=0.1

    #Broken CB-junction as part of chain-ester
    U_broken=20*factor
    tau_broken=1
    delta_broken=0.1


elif(junction=='et'):
        # PEG chain 
    U0=U_PEG
    tau0=tau_PEG
    delta0=delta_PEG

    # CB-junction 1,2 regiochemistry- ether and ester
    U11=1*factor
    U12=U11
    tau1=1
    delta1=0.1

##    # CB-junction 1,3 regiochemistry- ester and control
##    U2=1.2*factor
##    tau2=1
##    delta2=0.15

    #Broken CB-junction as part of chain-ether
    U_broken=3*factor
    tau_broken=1
    delta_broken=0.1

elif(junction=='ctrl'):
        # PEG chain 
    U0=U_PEG
    tau0=tau_PEG
    delta0=delta_PEG

##    # CB-junction 1,2 regiochemistry
##    U11=1*factor
##    U12=U11
##    tau1=1
##    delta1=0.1

    # CB-junction 1,3 regiochemistry- ester and control
    U2=0.5*factor
    tau2=1
    delta2=0.1

##    #Broken CB-junction as part of chain
##    U_broken=3*factor
##    tau_broken=1
##    delta_broken=0.1


 

rho  = 3
l0   = 1
prob = 1.0

E_b=1200
K  = 1.0
r0 = 0.0
##U0  = 1
##tau = 1
del_t = 0.005
erate = 10
lam_max = 30


tol=0.01
max_itr = 100000
write_itr = 10000
wrt_step = 5

