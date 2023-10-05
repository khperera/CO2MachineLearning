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
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import SmoothBivariateSpline


from CO2Fxn import fitCrossAndWLF
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
        self.R = 0.01
        self.Rmin = 0.00001
        self.H = 0.001
        self.angularVelocity = 1
        self.name = name
        self.a_n_list = []
        self.an = self.GenerateAn(10)
        self.CrossParams = [1.491095, 6.56460753e-1, 8754.40717e3, 25]
        self.Temperature = 100
        self.Pressure = 10
        self.TimeScale = 10800
        #eta will switch to numpy array later on
        #eta will be in the form of mx+b, where x is time. Guess m and b. m = 0.6,
        self.solubilityModel = self.importSolubilityModel("./SolubilityPS.xlsx")

        #solubility viscoisty models
        self.SolubilityViscModel = fitCrossAndWLF("./CombinedDataPS_p2.xlsx")
        self.SolubilityViscModel.fitDataTotal()

        #self.etaexp = dde.Variable(0.1, dtype=tf.float64)
        #self.eta = 10**self.etaexp
        self.Dexp = dde.Variable(-12, dtype=tf.float64)
        #self.b = dde.Variable(0.1, dtype=tf.float64)

        self.etaInit = 3000
        self.r = 4
        self.t = 1
        self.calcTorque = 167.55
        #be able to call system variables:



        self.location = "./NewTrainingData.csv"
        self.xData, self.torqueData = self.importData(self.location)

        #define eta. Should be a NP array

        #### setting up domain
        self.geom = dde.geometry.Interval(self.Rmin, self.R)
        self.timedomain = dde.geometry.TimeDomain(0.001, 1)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, self.timedomain)
        self.bc = self.generateBC()

        ### setting up model
        self.data = dde.data.TimePDE(
            self.geomtime,
            self.rheometerPlateCO2Diff,
            self.bc,
            num_domain=800,
            num_boundary=1000,
            #num_initial = 100,
            #solution = self.initialCondition
            #auxiliary_var_function=ex_func2,
        )

        self.external_trainable_variables = [self.Dexp]
        self.variable = dde.callbacks.VariableValue(
            self.external_trainable_variables, period=200, filename="variables1.dat"
        )


        self.net = dde.nn.FNN([2] + [80] * 5 + [1], "tanh", "Glorot uniform")
        self.net.apply_output_transform(self.transformFxn)
        self.model = dde.Model(self.data, self.net)



###############################################################################
##backend/ generation of system constants/math stuff


    def importData(self, location):
        Data = pd.read_csv(location)
        torque = [(Data["tau"])]
        #print(torque)
        time = [(Data["t"])]
        #print(time)

        r = [(Data["r"])]
        self.Temperature = Data["Temperature"].mean()
        self.Pressure = Data["Pressure"].mean()
        self.etaInit = Data["eta"].iloc[0]
        #print(r)
        #R, T = np.meshgrid(r, time)
        r = np.reshape(r, (-1, 1))
        t = np.reshape(time, (-1, 1))
        Torque =  np.reshape(torque, (-1, 1))

        return np.hstack((r,t)), Torque


    #makes zeros of the bessel function
    def GenerateAn(self,numberOfZeros):
        a_n = tf.constant(spe.jn_zeros(0, numberOfZeros), dtype=tf.float64)
        self.a_n_list =spe.jn_zeros(0, numberOfZeros)
        a_n = tf.reshape(a_n, [1, 1,-1])
        return a_n


    #defines the concentration of co2 analytically
    def concentrationSln(self,x):
        r = tf.reshape(x[:,0:1],[-1,-1,1])
        t = tf.reshape(x[:,1:2],[-1,-1,1])
        #print("r",r)
        #print("yoho", x[:,0:1])
        R = self.R
        D = 10**(self.Dexp)
        a_n = self.a_n_list
        #print("rda",R,D,a_n)

        print(a_n)
        #bessel_result = tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**(self.Dexp))*t/R**2)
        result_sum = 1-2*dde.backend(tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**(self.Dexp))*t/R**2), axis=2)

        print(2*tf.reduce_sum(tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**(self.Dexp))*t/R**2), axis=2))

        return 1-2*tf.reduce_sum(tf.math.special.bessel_j0(a_n*r/R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**(self.Dexp))*t/R**2), axis=2)

    def concentrationSln2(self,x):
        r = x[:,0:1]
        t = x[:,1:2]
        #print("r",r)
        #print("yoho", x[:,0:1])
        R = self.R
        D = 10**(self.Dexp)
        #a_n = self.an



        i = 0
        sum = 0
        #print(self.a_n_list)
        an1 = self.a_n_list[0]
        for a_n in self.a_n_list:
            if i == 0:
                an1 = a_n
                sum = 2*tf.math.special.bessel_j0(a_n * x[:,0:1] / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**self.Dexp)*x[:,1:2]*self.TimeScale/R**2)
                i = i + 1
            else:
                sum = sum + 2*tf.math.special.bessel_j0(a_n * x[:,0:1] / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**self.Dexp)*x[:,1:2]*self.TimeScale/R**2)

        #bessel_result = tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**(self.Dexp))*t/R**2)
        #result_sum = 1-2*tf.reduce_sum(tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(10**(self.Dexp))*t/R**2), axis=2)


        return 1-sum
        #return 1-2*tf.math.special.bessel_j0(an1 * x[:,0:1] / R)/(tf.math.special.bessel_j1(an1)*an1)*tf.exp(-an1**2*((10**self.Dexp))*x[:,1:2]*self.TimeScale/R**2)

    #function that defines viscosity as function of concentration
    def concentrationToViscosity(self,concentration, zeroShear):
        return concentration + zeroShear


###############################################################################
#defining boundaries


    def boundary(self,_, on_initial):
        return on_initial

    def inside(self,x, on_boundary):
        return on_boundary and np.equal(x[0], self.Rmin)
    def outside(self,x, on_boundary):
        return on_boundary and np.equal(x[0],self.R)

    def generateBC(self):
        bc = dde.icbc.DirichletBC(self.geomtime, lambda x: 0, self.inside, component=0)
        ic = dde.icbc.IC(self.geomtime, self.initialCondition,
            lambda _, on_initial: on_initial
            )
        #bc2 = dde.icbc.DirichletBC(self.geomtime, lambda x: self.calcTorque, self.outside)
        bc3 = dde.icbc.NeumannBC(self.geomtime, lambda x: 0, self.inside, component=0)

        #data from csv

        observation_BC =  dde.icbc.PointSetBC(self.xData, self.torqueData)

        return [bc,bc3, observation_BC]


    def initialCondition(self,x):
        return 2*3.14*self.etaInit*self.angularVelocity*x[:, 0:1]**3/3/self.H
        #newSim.runBFGS(20000,[100,1,1,100])

#newSim.plotEta()




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

        #eta = 5




        #eqn = ()
        return (Tau_r*self.H/(2*3.1415*(self.etaFunction(x))*self.angularVelocity*self.R**3)-((r/self.R)**2))
        #return (eta*dVtheta_zz)
        #return (eta*dVtheta_r_r)


    def rheometerPlateCO2Diff(self,x,y):
        Tau =  y[:, 0:1]
        #x1 =tf.reshape(x, [-1,-1,1])

        r = x[:,0:1]
        t = x[:, 1:2]

        Tau_r =  dde.grad.jacobian(y,x,i=0)
        #viscosityMod = self.etaFunction2(x)







        #eqn = ()
        return (Tau_r*self.H/(2*3.1415*(self.etaFunction2(x))*self.angularVelocity*self.R**3)-((r/self.R)**2))
        #return (eta*dVtheta_zz)
        #return (eta*dVtheta_r_r)

    def transformFxn(self, x, y):
        viscosityMod = self.etaFunction2(x)
        res = self.H/(2*3.1415*(viscosityMod)*self.angularVelocity*self.R**3)
        return ( y/res)



###############################################################################
#def model running and saving models


    #BFGS, weights should be given in array format
    def runBFGS(self, iterations, weights):
        self.model.compile("L-BFGS",loss_weights=weights,external_trainable_variables=self.external_trainable_variables)
        #model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
        dde.optimizers.config.set_LBFGS_options(
        maxcor=100,
        ftol=1.0e-50,
        gtol=1.0e-35,
        maxiter=60000,
        maxfun=50000,
        maxls=50,
        )

        self.losshistory, self.train_state = self.model.train(iterations=iterations,callbacks = [self.variable])

    #BFGS, weights should be given in array format, learning rate given as float
    def runAdam(self, iterations, weights, learningRate):

        self.model.compile(
        "adam", lr=learningRate,loss_weights=weights,external_trainable_variables=self.external_trainable_variables
        )
        self.losshistory, self.train_state = self.model.train(iterations=iterations,callbacks = [self.variable])

    def savePlot(self):
        dde.saveplot(self.losshistory, self.train_state, issave=True, isplot=True)


###############################################################################
#debugging stuff
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
###############################################################################
#data for models

    #returns viscosity given cross parameter models and shear rate. Done for 0 Bar. Takes in shift factor too
    def crossModel(self,x, crossParams, shiftFactor):
        k,n,viscoZero, viscoInf = crossParams
        differenceInVisc = viscoZero - viscoInf
        shearRate = x[:,0:1]*self.angularVelocity/self.H*shiftFactor
        return (differenceInVisc/(1+(k*shearRate)**n)+viscoZero)/shiftFactor


    #linear test function of eta. See if we can guess b and m ?
    #proven we can do this.
    def etaFunction(self,x):

        return x[:,1:2]*self.m+self.b


    def etaFunction2(self,x):
        MaxSolubility = self.solubilityModel.ev(self.Temperature ,self.Pressure)
        #given nondimensionalConcentration
        concentration = self.concentrationSln2(x)
        shiftFactorT = self.SolubilityViscModel.runWLFModelT(x, self.Temperature)
        shiftFactorS = self.SolubilityViscModel.runWLFModelS(x, concentration)
        #give shear rate, get viscosity
        ViscosityNaught = self.SolubilityViscModel.CrossModelOut(x[:,0:1]*self.angularVelocity/self.H)
        return ViscosityNaught*shiftFactorT*shiftFactorS

    #import data from excel. Returns a model that takes in pressure and temperature. This will be run once to get max solubility for a run.
    def importSolubilityModel(self, locationOfData):
        DataForPS = pd.read_excel(locationOfData, header = [0])
        x= DataForPS["Pressure (MPa)"]
        y=DataForPS["Temperature (K)"]
        z=DataForPS["Solubility"]
        Model = SmoothBivariateSpline(x,y,z,kx=2,ky=2, s = 0.0001,eps=0.02)
        return Model

    #returns a solubility given a pressure and temeprature.
    def estimateSolubility(self,  Temp, Pressure):
        pass

    #shift factor for solubility data.
    def defImportShiftFactor(self, WLFParams, solubility):
        C1, C2 = WLFParams
        return 10**(-C1*(solubility-0)/(C2+solubility-0))




#sixTBar.plotCompareData2("loc")

newSim = CO2Sim("file")

newSim.runAdam(8000,[1,1,1,100], 0.005)
newSim.runBFGS(10000,[1,1,1,100])
newSim.savePlot()
