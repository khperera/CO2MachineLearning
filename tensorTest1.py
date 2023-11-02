#simple rheometer flow

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
#import deepxde as dde
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import special as spe
from CO2Fxn import fitCrossAndWLF
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import SmoothBivariateSpline
import scipy.integrate as integrate
import tensorflow_probability as tfp
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC
print(tf.__version__)

#constants
#r = [1,2,3,4,5,6,7,8,9]
#t = [1,2,3,4,5]
D = 1e-9
R = 0.01
T = 10000
angularVelocity = 1
H = 0.001
Temperature = 120
Pressure = 80






# Define tensors for a_n and r (assuming 1D tensors for demonstration)
#a_n = tf.constant([0.1, 0.2, 0.3, 0.4,4,5,6], dtype=tf.float32)
numberofZeroes = 500
a_n = tf.constant(spe.jn_zeros(0, numberofZeroes), dtype=tf.float32)
r1 = np.arange(0.0001,R,0.0005)
t1 = np.linspace(1,np.log10(T),20)

print(r1)
r = tf.constant(r1, dtype=tf.float32)
t = tf.constant(t1, dtype=tf.float32)
# Reshape to enable broadcasting
a_n = tf.reshape(a_n, [-1, 1,1])  # Shape becomes [M, 1]
r = tf.reshape(r, [1, -1,1])      # Shape becomes [1, N]
t =tf.reshape(t, [1, 1,-1])

# Compute bessel_j0(a_n * r / 5)

















bessel_result = tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*D*10**t/R**2)
#                sum = 2*tf.math.special.bessel_j0(a_n * x[:,0:1] )/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*(self.diffusionModel())*x[:,1:2]*self.TimeScale/R**2)




# Sum along the axis corresponding to a_n (axis=0)
result_sum = tf.reduce_sum(bessel_result, axis=0)


# Create a MultiIndex using the r1 and t1 arrays
index = pd.MultiIndex.from_product([r1, t1], names=['r', 't'])
# Flatten the bessel_result_sum array to match the MultiIndex
bessel_result_1d = tf.reshape(result_sum, [-1])
# Convert the tensor to a NumPy array
bessel_result_1d_np = 1-2*np.abs(bessel_result_1d.numpy())
# Create the DataFrame
df = pd.DataFrame({'Bessel_Sum': bessel_result_1d_np}, index= index)
#print(df)















def importSolubilityModel(locationOfData):
    DataForPS = pd.read_excel(locationOfData, header = [0])
    x= DataForPS["Pressure (MPa)"]/0.1
    y=DataForPS["Temperature (K)"]-273
    z=DataForPS["Solubility"]
    Model = SmoothBivariateSpline(x,y,z,kx=2,ky=2, s = 0.0001,eps=0.02)
    return Model





#tf.print(result_sum, output_stream=sys.stdout)

#fig1 = sns.scatterplot(data = df, x = "r", y = "Bessel_Sum", hue = "t")
#plt.show()




solubilityModel = importSolubilityModel("./SolubilityPS.xlsx")
SolubilityViscModel = fitCrossAndWLF("./CombinedDataPS_p2.xlsx")
SolubilityViscModel.fitDataTotal()
MaxSolubility = solubilityModel.ev(Pressure,Temperature)
print(MaxSolubility, "maxsol")
shiftFactorT = SolubilityViscModel.runWLFModelT(r, Temperature)
print("shiftFactorT", shiftFactorT)

df["ShiftFactorS"] = SolubilityViscModel.runWLFModelS(df["Bessel_Sum"], MaxSolubility*df["Bessel_Sum"])

df = df.reset_index()

#fig1 = sns.scatterplot(data = df, x = "r", y = "ShiftFactorS", hue = "t")
#plt.show()

df["ViscosityNaught"] = SolubilityViscModel.CrossModelOut(df["r"]*angularVelocity/H*shiftFactorT*df["ShiftFactorS"])
df["ViscosityFinal"] = df["ViscosityNaught"] *df["ShiftFactorS"]*shiftFactorT
constants = 2*3.1415*angularVelocity/H

df["Dtaudr"] = df["r"]**2*constants
df["t_shift"] = df["t"]/np.log10(100000)

r2 = []
t2 = []
c2 = []
def viscosityFromVariables(R, Temperature, Time):
    numberofZeroes = 50
    a_n = tf.constant(spe.jn_zeros(0, numberofZeroes), dtype=tf.float64)
    r1 = R
    #t1 = np.linspace(1,np.log10(Time),1)
    t1 = Time

    BigR = tf.constant(0.01, dtype=tf.float64)

    #print(r1)
    r = tf.constant(r1, dtype=tf.float64)
    t = tf.constant(t1, dtype=tf.float64)
    # Reshape to enable broadcasting
    a_n = tf.reshape(a_n, [-1, 1,1])  # Shape becomes [M, 1]
    r = tf.reshape(r, [1, -1,1])      # Shape becomes [1, N]
    t =tf.reshape(t, [1, 1,-1])

    D = tf.constant(10**-9, dtype=tf.float64)



    bessel_result = tf.math.special.bessel_j0(a_n * r / BigR)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*D*t/BigR**2)

    #print("bessel_result",bessel_result)

    MaxSolubility = solubilityModel.ev(Pressure,Temperature)

    result_sum = 1- 2*tf.reduce_sum(bessel_result, axis=0)
    shiftFactorS_calc = SolubilityViscModel.runWLFModelS(result_sum, MaxSolubility*result_sum)

    shiftFactorT_calc = SolubilityViscModel.runWLFModelT(r, Temperature)
    viscosityFinal_calc = SolubilityViscModel.CrossModelOut(r*angularVelocity/H*shiftFactorT_calc*shiftFactorS_calc)*shiftFactorT_calc*shiftFactorS_calc
    #print("shiftFactorT_calc",shiftFactorT_calc)

    return viscosityFinal_calc.numpy()

def tauDerivative(R,Temperature, Time):
    constants = 2*3.1415*angularVelocity/H
    tauDer = constants*viscosityFromVariables(R,Temperature,Time)*R**2
    #print("tauDer",tauDer)
    return tauDer



def tauIntegration(Time, R, Temperature):
    #print(sampleSpace)
    #Time, R, Temperature = sampleSpace
    #print("R,T", R, Time)
    tau_integrate,residual = integrate.quad_vec(lambda r: tauDerivative(r, Temperature, Time), 0, R)
    #print(tau_integrate)
    #tau_integrate,residual = integrate.quad(lambda r: r*Time*Temperature, 0, R)

    return tau_integrate

def tauIntegration2(Time, R, Temperature):
    tauInter = np.vectorize(tauIntegration)
    integrated = tauInter(Time,R,Temperature)

    return integrated


def importData(location):
    Data = pd.read_csv(location)
    torqueScale = (Data["tau"]).max()
    #self.TimeScale = Data["t1"].max()
    #self.logTimeScale = np.log10(self.PresetTimeScale)
    #print("Timescale", self.TimeScale)
    #print("torquescale", self.torqueScale)
    Data["tau_scale"] = (Data["tau"])/torqueScale
    #Data["tau_scale"] = np.log(Data["tau_scale"]+self.fitSketcher)
    Data["time_scale"] = np.log10(Data["t1"])/5





    #estimate shift shiftFactorS
    #self.shiftFactorT = self.SolubilityViscModel.runWLFModelT(x, self.Temperature)
    #print("shift factor T:", self.shiftFactorT )




    return Data



def calculateStress(R, Temperature, Time):
    numberofZeroes = 50
    a_n = tf.constant(spe.jn_zeros(0, numberofZeroes), dtype=tf.float64)
    #r1 = R


    r1 = np.arange(R/100,R,R/100)

    #t1 = np.linspace(1,np.log10(Time),1)
    #t1 = Time

    BigR = tf.constant(R, dtype=tf.float64)

    #print(r1)
    r1 = tf.constant(r1, dtype=tf.float64)
    t1 = tf.constant(Time, dtype=tf.float64)
    # Reshape to enable broadcasting
    a_n = tf.reshape(a_n, [-1, 1,1])  # Shape becomes [M, 1]
    r1 = tf.reshape(r1, [1, -1,1])      # Shape becomes [1, N]
    t =tf.reshape(t1, [1, 1,-1])

    D = tf.constant(10**-9, dtype=tf.float64)



    bessel_result = tf.math.special.bessel_j0(a_n * r1 / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*D*t/R**2)

    #print("bessel_result",bessel_result)

    MaxSolubility = solubilityModel.ev(Pressure,Temperature)

    result_sum = 1- 2*tf.reduce_sum(bessel_result, axis=0)
    shiftFactorS_calc = SolubilityViscModel.runWLFModelS(result_sum, MaxSolubility*result_sum).numpy()

    shiftFactorT_calc = SolubilityViscModel.runWLFModelT(r1, Temperature)
    viscosityFinal_calc = SolubilityViscModel.CrossModelOut(r1*angularVelocity/H*shiftFactorT_calc*shiftFactorS_calc)*shiftFactorT_calc*shiftFactorS_calc

    constants = 2*3.1415*angularVelocity/H
    tauDer = constants*viscosityFinal_calc*r1**2
    Stress = tfp.math.trapz(tauDer,axis = 1)
    return Stress.numpy(),result_sum,r1,t



    #print("shiftFactorT_calc",shiftFactorT_calc)








Temperature = 150
Pressure = 80
timeDim = 15
rDim = 1
TDim = 1
timeRange = np.logspace(0,4,timeDim).reshape((timeDim,1,1))
#timeRange = np.logspace(0,5,timeDim).reshape((timeDim,1,1))
rRange = np.linspace(0.01,0.01,rDim).reshape((1,rDim,1))
Temperature = np.linspace(130,130,TDim).reshape((1,1,TDim))
tspace, rspace, Tspace = np.meshgrid(timeRange, rRange, Temperature, indexing = "xy")

Stress,concentration,r1,t1 = calculateStress(0.01,130,timeRange)
print(concentration.numpy().max())





#tau = np.array(tauIntegration2(tspace,rspace,Tspace))
##print(tspace.shape)
#print(tau.shape)







#ataFrameData["tau"] = tauIntegration(dataFrameData["r"],dataFrameData["Temperature"],dataFrameData["TimeSpace"])
ExperimentalData = importData("./TrainingData/"+"80-150"+".csv")



fig = plt.figure()
ax = fig.add_subplot()

ax.scatter(np.log10(timeRange)/np.log10(100000), Stress/Stress.max(), color='blue')
ax.scatter( ExperimentalData["time_scale"],  ExperimentalData["tau_scale"], color='red')

ax.set_xlabel('time')
ax.set_ylabel('stresss')

#plt.xscale("log")

plt.show()










#fig1 = sns.scatterplot(data = df, x = "t_shift", y = "ViscosityFinal", hue = "r")
name = "./figures/"+str(D)+"-P"+str(Pressure)+"-T"+str(Temperature)+"viscosityFinal.png"
#plt.savefig(name, dpi = 1000)
#plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["r"] , df["t_shift"], df["ShiftFactorS"] , color='blue')

#X, Y = np.meshgrid(x_range, y_range)
#Z = Model.ev(X,Y)
#ax.plot_surface(X,Y,Z, color='red', alpha=0.5)
ax.set_xlabel('r')
ax.set_ylabel('t')
ax.set_zlabel('viscosity')
plt.show()




















"""
## zeros of bessel fxn
#bessel fxn
numberofZeroes = 5
an = spe.jn_zeros(0, numberofZeroes)
print(an)
an = tf.reshape(an, [1, 5])







C = 1 - 2*tf.reduce_sum(tf.math.special.bessel_j0(an*r/5)/(an*tf.math.special.bessel_j1(an))*tf.exp(-an**2*D*t/5**2),0)




tf.print(C, output_stream=sys.stdout)
"""
