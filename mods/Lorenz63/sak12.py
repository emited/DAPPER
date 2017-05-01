# Reproduce results from Table 1 of Sakov et al "iEnKF" (2012)

from common import *

from mods.Lorenz63.core import step, dfdx

m = 3
p = m

t = Chronology(0.01,dkObs=25,T=4**5,BurnIn=4)

f = {
    'm'    : m,
    'model': step,
    'jacob': dfdx,
    'noise': 0
    }

mu0 = array([1.509, -1.531, 25.46])
X0 = GaussRV(C=2,mu=mu0)

h = {
    'm'    : p,
    'model': Id_op(),
    'jacob': Id_mat(m),
    'noise': 2, # abbrev GaussRV(C=CovMat(2*eye(p)))
    }

other = {'name': os.path.relpath(__file__,'mods/')}

setup = TwinSetup(f,h,t,X0,**other)


####################
# Suggested tuning
####################
#from mods.Lorenz63.sak12 import setup ##################### Expected RMSE_a:
#cfgs += Climatology()  # no tuning!                       # 7.6
#cfgs += OptInterp()    # no tuning!                       # 1.25
#cfgs += Var3D(infl=0.9)# tuning not strictly required     # 1.03 
#cfgs += ExtKF(infl=90) # some inflation tuning needed     # 0.87
#cfgs += EnKF('Sqrt',   N=3 ,  infl=1.30)                  # 0.82
#cfgs += EnKF('Sqrt',   N=10,  infl=1.02,rot=True)         # 0.63
#cfgs += EnKF('PertObs',N=500, infl=0.95,rot=False)        # 0.56
#cfgs += EnKF_N(        N=10,            rot=True)         # 0.54
#cfgs += iEnKF('Sqrt',  N=10,  infl=1.02,rot=True,iMax=10) # 0.31
#cfgs += PartFilt(      N=100 ,reg=2.4,NER=0.3)            # 0.38
#cfgs += PartFilt(      N=800 ,reg=0.9,NER=0.2)            # 0.28
#cfgs += PartFilt(      N=4000,reg=0.7,NER=0.05)           # 0.27
#cfgs += PFD(xN=1000,   N=30  ,reg=0.7,NER=0.2,Qs=2)       # 0.56
