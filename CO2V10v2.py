###############################################################################

### basic imports
"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
import deepxde as dde
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import special as spe
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC

###############################################################################

dde.config.set_default_float("float64")
dde.config.real.set_float64()




###############################################################################
#make class for actual PDE
class CO2Sim():


    def __init__(self,name):

        ## system variables
        self.R = 1
        self.Rmin = 0.0001
        self.H = 0.05
        self.angularVelocity = 1
        self.name = name
        self.an = self.GenerateAn(10)
        #eta will switch to numpy array later on
        self.eta = 4
        self.etaInit = 4
        self.r = 4
        self.t = 1
        #be able to call system variables:

        #define eta. Should be a NP array

        #### setting up domain
        self.geom = dde.geometry.Interval(self.Rmin, self.R)
        self.timedomain = dde.geometry.TimeDomain(0, 2)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)
        self.bc = self.generateBC()

        ### setting up model
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.rheometerPlate,
            self.bc,
            num_domain=800,
            num_boundary=100,
            num_initial = 100
            #solution = slnfuc
            #auxiliary_var_function=ex_func2,
        )
        print(self.eta)

        self.net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
        self.net.apply_output_transform(self.transformFxn)
        self.model = dde.Model(self.data, self.net)



###############################################################################
##backend/ generation of system constants/math stuff

    #makes zeros of the bessel function
    def GenerateAn(self,numberOfZeros):
        a_n = tf.constant(spe.jn_zeros(0, numberOfZeros), dtype=tf.float64)
        a_n = tf.reshape(a_n, [1, 1,-1])
        return a_n


    #defines the concentration of co2 analytically
    def concentrationSln(self,r,t,R,a_n,D):
        #r = tf.reshape(r1,[-1,-1,1])
        #t = tf.reshape(t1,[-1,-1,1])
        bessel_result = tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*D*t/R**2)
        result_sum = 1-2*tf.reduce_sum(bessel_result, axis=2)



        return result_sum

    #function that defines viscosity as function of concentration
    def concentrationToViscosity(self,concentration, zeroShear):
        return concentration + zeroShear

    #def dummyEta()

###############################################################################
#defining boundaries


    def boundary(self,_, on_initial):
        return on_initial

    def inside(self,x, on_boundary):
        return on_boundary and np.equal(x[0], self.Rmin)
    def outside(self,x, on_boundary):
        return on_boundary and np.equal(x[0],R)

    def generateBC(self):
        bc = dde.icbc.DirichletBC(self.geomtime, lambda x: 0, self.inside, component=0)
        #bc = dde.icbc.DirichletBC(self.geom, lambda x: 0, self.inside)
        #bc3 = dde.icbc.NeumannBC(self.geom, lambda x: 0, self.inside)
        ic = dde.icbc.IC(self.geomtime, self.initialCondition,
            lambda _, on_initial: on_initial
            )
        #bc2 = dde.icbc.DirichletBC(geomtime, lambda x: torqueOut, outside)
        bc3 = dde.icbc.NeumannBC(self.geomtime, lambda x: 0, self.inside, component=0)
        return [bc,bc3]


    def initialCondition(self,x):
        return 2*3.14*self.etaInit*self.angularVelocity*x[:, 0:1]**3/3/self.H

###############################################################################
#def system
    #rheometer plate
    def rheometerPlate(self,x,y):
        Tau =  y[:, 0:1]
        #x1 =tf.reshape(x, [-1,-1,1])

        r = x[:,0:1]
        t = x[:, 1:2]
        self.r = r
        self.t = t
        Tau_r =  dde.grad.jacobian(y,x,i=0)

        eta = self.etaInit+t/2*100
        #eta = 5




        #eqn = ()
        return (Tau_r*self.H/(2*3.1415*(self.etaInit)*self.angularVelocity*self.R**3)-((r/self.R)**2/self.R))
        #return (eta*dVtheta_zz)
        #return (eta*dVtheta_r_r)

    def transformFxn(self, x, y):
        res = self.H/(2*3.1415*(self.etaInit)*self.angularVelocity*self.R**3)
        return (res * y)
###############################################################################

#def model running

    #BFGS, weights should be given in array format
    def runBFGS(self, iterations, weights):
        self.model.compile("L-BFGS",loss_weights=weights)
        #model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
        dde.optimizers.config.set_LBFGS_options(
        maxcor=100,
        ftol=1.0e-35,
        gtol=1.0e-35,
        maxiter=50000,
        maxfun=50000,
        maxls=50,
        )

        self.losshistory, self.train_state = self.model.train(iterations=iterations)

    #BFGS, weights should be given in array format, learning rate given as float
    def runAdam(self, iterations, weights, learningRate):

        self.model.compile(
        "adam", lr=learningRate,loss_weights=weights
        )
        self.losshistory, self.train_state = self.model.train(iterations=iterations)

    def savePlot(self):
        dde.saveplot(self.losshistory, self.train_state, issave=True, isplot=True)
###############################################################################
    def plotEta(self):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.r.numpy(), self.t.numpy(), self.eta.numpy(), color='blue')


        ax.set_xlabel('radius')
        ax.set_ylabel('time')
        ax.set_zlabel('Viscosity')
        plt.show()
    def print(self):
        tf.print(self.eta, output_stream=sys.stdout)
        print("HELLO",self.eta)



newSim = CO2Sim("file")
newSim.runAdam(10000,[100,1,1], 0.005)
newSim.print()
newSim.runBFGS(10000,[100,1,1])
newSim.print()
#newSim.runAdam(10000,[100,1,1,100], 0.005)
#newSim.runBFGS(20000,[100,1,1,100])
#newSim.plotEta()
newSim.savePlot()
