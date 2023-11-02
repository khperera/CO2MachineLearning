#simple rheometer flow

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
#import deepxde as dde
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
#from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import pandas as pd
#from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import special as spe
from CO2Fxn import fitCrossAndWLF
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import SmoothBivariateSpline
import scipy.integrate as integrate
import scipy
#import tensorflow_probability as tfp
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC
#print(tf.__version__)

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
#shiftFactorT = SolubilityViscModel.runWLFModelT(r, Temperature)


def viscosityFromVariables(R, Temperature, Time):
    numberofZeroes = 50
    a_n = tf.constant(spe.jn_zeros(0, numberofZeroes), dtype=tf.float64)
    r1 = R
    #t1 = np.linspace(1,np.log10(Time),1)
    t1 = Time

    #BigR = tf.constant(0.01, dtype=tf.float64)
    BigR = 0.01
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
    Data["time_scale"] = np.log10(Data["t1"])





    #estimate shift shiftFactorS
    #self.shiftFactorT = self.SolubilityViscModel.runWLFModelT(x, self.Temperature)
    #print("shift factor T:", self.shiftFactorT )




    return Data


#where Router is the Total Radius of the system
def calculateStress(R, Temperature, Time,D1, Pressure,Router):
    numberofZeroes = 50
    a_n = (spe.jn_zeros(0, numberofZeroes))
    #r1 = R


    r1 = np.arange(R/100,R,R/100)

    #t1 = np.linspace(1,np.log10(Time),1)
    #t1 = Time

    BigR = R
    H = 0.001

    #print(r1)
    r1 = r1
    t1 = Time
    # Reshape to enable broadcasting
    a_n = np.reshape(a_n, [-1, 1,1])  # Shape becomes [M, 1]
    r1 = np.reshape(r1, [1, -1,1])      # Shape becomes [1, N]
    t = np.reshape(t1, [1, 1,-1])

    D = 10**D1



    bessel_result = scipy.special.j0(a_n * r1 / Router)/(scipy.special.j1(a_n)*a_n)*np.exp(-a_n**2*D*t/Router**2)

    #print("bessel_result",bessel_result)

    MaxSolubility = solubilityModel.ev(Pressure,Temperature)

    result_sum = 1- 2*np.sum(bessel_result, axis=0)
    shiftFactorS_calc = SolubilityViscModel.runWLFModelS(result_sum, MaxSolubility*result_sum)

    shiftFactorT_calc = SolubilityViscModel.runWLFModelT(r1, Temperature)
    viscosityFinal_calc = SolubilityViscModel.CrossModelOut(r1*angularVelocity/H*shiftFactorT_calc*shiftFactorS_calc)*shiftFactorT_calc*shiftFactorS_calc
    #print(viscosityFinal_calc.min(), viscosityFinal_calc.max())
    constants = 2*3.1415*angularVelocity/H
    #print(constants)
    tauDer = constants*viscosityFinal_calc*r1**3
    Stress = np.trapz(tauDer,axis = 1, dx=R/100)
    return Stress,result_sum,r1,t



    #print("shiftFactorT_calc",shiftFactorT_calc)
#inputvars are [dexp,shift]. Time is numpy array, experimental data is numpy array, should be normalized
def minimizeFxn(InputVars, Time, ExperimentalData, Temperature,Pressure, Radius,MaxStress,Router):
    
    minTime = Time.min()
    maxTime = Time.max()
    TimeSpace = np.logspace(np.log10(minTime),np.log10(maxTime), 50)
    ExperimentalData = InputVars[2]+ExperimentalData
    
    CalculatedStress,conc,r1,t1 =  calculateStress(Radius,Temperature,Time,InputVars[0],Pressure,Router)
    CalculatedStress = CalculatedStress.squeeze()
    
    SplineCalculated = scipy.interpolate.UnivariateSpline(np.log10(Time), CalculatedStress/MaxStress)
    
    SplineExperimental = scipy.interpolate.UnivariateSpline(np.log10(Time), ExperimentalData)
    
    SplinedCal = SplineCalculated(np.log10(TimeSpace))
    SplinedExp = SplineExperimental(np.log10(TimeSpace))
    
    #DefinedTimeRange = numpy.logspace(np.log10(minTime),np.log10(maxTime),50)
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.scatter( np.log10(Time),  CalculatedStress/MaxStress, color='black')
    
    ax.scatter( np.log10(Time), ExperimentalData * InputVars[1], color='purple')
    ax.scatter( np.log10(TimeSpace),SplinedCal, color='red')
    
    ax.scatter( np.log10(TimeSpace), SplinedExp* InputVars[1], color='orange')
    

    ax.set_xlabel('time')
    ax.set_ylabel('stresss')

    #plt.xscale("log")

    plt.show()
    """
    
    Difference = np.subtract(SplinedCal,SplinedExp*InputVars[1])**2#*(np.log10(TimeSpace)-3)**4
    TotalDiff = np.sum(Difference)
    return TotalDiff
    
    
    
    
    


def plotOptimizer(expData, Temperature, Pressure):
    timeDim = 50
    r = 0.01
    Router = 0.011
    rDim = 1
    TDim = 1
    timeRange = np.logspace(0,5,timeDim).reshape((timeDim,1,1))
    #timeRange = np.logspace(0,5,timeDim).reshape((timeDim,1,1))
    rRange = np.linspace(r,r,rDim).reshape((1,rDim,1))
    Temperature = np.linspace(Temperature,Temperature,TDim).reshape((1,1,TDim))
    
    
    
    Time = expData["t1"].to_numpy()
    ExpStressData = (expData["tau"]/expData["tau"].max()).to_numpy()
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    

    colors = ["red","yellow","orange","green","blue"]
    number = [0,1,2,3]
    DConst = [-8,-9,-10,-11]
    MaxStress = 0
    for num in number:
        Stress,concentration,r1,t1 = calculateStress(r,Temperature,timeRange,DConst[num],Pressure,Router)
        if Stress.max()>MaxStress:
            MaxStress = Stress.max()
        ax.scatter(np.log10(timeRange), Stress/Stress.max(), color=colors[num])
        
    bnds = ((-11, -7), (0.9,2), (0,0.2))
    xo = [-9,0.95,0]
    argu = (Time,ExpStressData,Temperature,Pressure,r,MaxStress,Router)
    optimizedVals = scipy.optimize.minimize(minimizeFxn,xo,args = argu,bounds = bnds)
    print(optimizedVals.x)
    
    
    Dexp = optimizedVals.x[0]
    Stress1,conc,radius,time = calculateStress(r,Temperature, Time, Dexp, Pressure,Router)
    
    ax.scatter( np.log10(Time),  Stress1/Stress.max(), color='black')
    
    ax.scatter( np.log10(Time),  (ExpStressData+optimizedVals.x[2])* optimizedVals.x[1], color='purple')

    ax.set_xlabel('time')
    ax.set_ylabel('stresss')

    #plt.xscale("log")

    plt.show()
 
    
    return Dexp







TPlist = [(100,120),(100,130),(20,150),(40,150),(80,120),(80,130),(80,150)]
#TPlist = [(100,120)]

T = 120
P = 80
DList = []
#list of samples 100-120,100-130,20-150, 40-150, 80-120, 80-130, 80-150
#ataFrameData["tau"] = tauIntegration(dataFrameData["r"],dataFrameData["Temperature"],dataFrameData["TimeSpace"])
for val in TPlist:
    P,T = val
    
    ExperimentalData = importData("./TrainingData/"+str(P)+"-"+str(T)+".csv")
    D = plotOptimizer(ExperimentalData, T,P)
    DList.append((T,P,D))

DataFrameList = pd.DataFrame(data = DList, columns = ["Temp","Pressure", "D"])
print(DataFrameList)

figure1 = sns.relplot(data = DataFrameList, x = "Temp", y = "D", hue = "Pressure")
#figure1.set()
plt.show()
"""
fig = plt.figure()
ax = fig.add_subplot()

colors = ["red","yellow","orange","green","blue"]
number = [0,1,2,3]
DConst = [-8,-9,-10,-11]
StressMin = 1
for num in number:
    Stress,concentration,r1,t1 = calculateStress(0.01,T,timeRange,DConst[num],P)
    ax.scatter(np.log10(timeRange)/np.log10(100000), Stress/Stress.max(), color=colors[num])




ax.scatter( ExperimentalData["time_scale"],  ExperimentalData["tau"]/ExperimentalData["tau"].max()*0.9, color='purple')

ax.set_xlabel('time')
ax.set_ylabel('stresss')

#plt.xscale("log")

plt.show()

"""




























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
