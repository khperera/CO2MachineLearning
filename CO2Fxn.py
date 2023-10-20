#analyzing CO2 data, start with a class for simulataneous fitting of cross and WLF parameters.

#Hi
import csv
import pandas as pd
import glob
import os
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from scipy.interpolate import SmoothBivariateSpline
from scipy import special as spe
import seaborn as sns
import math
# Apply the default theme

class fitCrossAndWLF():
    def __init__(self,fileLocation):
        self.filelocation = fileLocation
        self.ReadData = self.loadData()
        self.SpecificDataSet = pd.DataFrame()
        self.mostRecentParams = []
        self.SolubilityModel = self.importSolubilityModel("./SolubilityPS.xlsx")
        self.interpolateSolubilityData()


        #physics specificVariables
        self.TRef = 120
        self.PRef = 0
        self.SRef = 0
        self.mini = 199999999999999999999999999
        self.Temperature = 100

############################################################################################################################################################################
#preprocessing

    #loads data from excel file into a pandas dataframe.
    def loadData(self):
        ReadData = pd.read_excel(self.filelocation, header = [0])

        return ReadData

    #returns solubility model for use later
    def importSolubilityModel(self,location):
        DataForPS = pd.read_excel(location, header = [0])
        #print(DataForPS)

        x= DataForPS["Pressure (MPa)"]/0.1
        y=DataForPS["Temperature (K)"]-273
        z=DataForPS["Solubility"]
        Model = SmoothBivariateSpline(x,y,z,kx=2,ky=2, s = 0.0001,eps=0.02)

        """

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, color='blue')
        x_range = np.linspace(0, 400, 50)
        y_range = np.linspace(50, 200,50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = Model.ev(X,Y)
        ax.plot_surface(X,Y,Z, color='red', alpha=0.5)
        ax.set_xlabel('Pressure')
        ax.set_ylabel('Temperature')
        ax.set_zlabel('Solubility')
        plt.show()
        """


        return Model

    def interpolateSolubilityData(self):
        self.ReadData["Solubility"] = self.SolubilityModel.ev(self.ReadData["Pressure"],self.ReadData["Temperature"])
        #self.ReadData.to_csv("F:/Downloads/Solubility.csv")
        return

############################################################################################################################################################################


    #Takes a selection of data and tries to fit the WLF model and cross. Returns parameters
    #uses scipy optimize to find parameters
    def fitWLFCrossModel(self,sliceOfData):
        self.SpecificDataSet = sliceOfData

        #initial guess of parameters C1,C2, k, n zeroS, infS
        parameters = [5, 100, 1, 1, 500, 10]

        #Bounds, general bounds
        Bounds = ((0.001,3000),(0.00001,3000),(0.001,5),(0.0001,5),(10,10000),(1,1000))
        WLFCrossModel

        optimizationResult = minimize(self.WLFCrossModel, parameters, bounds = Bounds, method='Nelder-Mead', tol=1e-12)

        print(list(optimizationResult.x), optimizationResult.success, optimizationResult.message)
        self.mostRecentParams = optimizationResult.x
        return self.mostRecentParams


    #Takes a selection of data and tries to fit the WLF model and cross. Returns parameters
    #uses scipy optimize to find parameters
    def fitWLFCrossModelPressure(self,sliceOfData):
        self.SpecificDataSet = sliceOfData

        #initial guess of parameters C1,C2, k, n zeroS, infS
        parameters = [7, 183, 2.2,59,1.5,0.54, 3500, 2]

        #Bounds, general bounds
        Bounds = ((1,1000),(1,50000),(0.1,50000),(0.1,5000),(0.01,4),(0.01,1),(1,5000),(1,1000))


        optimizationResult = minimize(self.WLFCrossModelPressure, parameters,options = {"maxiter":50000, "ftol":1e-24}, bounds = Bounds, method='L-BFGS-B', tol=1e-12)

        #print(list(optimizationResult.x), optimizationResult.success, optimizationResult.message)
        self.mostRecentParams = optimizationResult.x
        return self.mostRecentParams





############################################################################################################################################################################
#visualizaiton


    def plotSolubilityData(self):
        pass




    #graphs experimental vs fit data for WLF model
    def plotCompareData(self,saveLoc):
        params = self.fitData()
        DataSetReal = self.SpecificDataSet
        print(params)
        DataSetReal["at"] = self.WLFModel(DataSetReal["Angular Frequency"],DataSetReal["Temperature"],self.TRef,params[0],params[1])
        print(DataSetReal["at"])
        DataSetReal["viscShift"] = DataSetReal["Complex viscosity"]/DataSetReal["at"]
        DataSetReal["freqShift"] = DataSetReal["Angular Frequency"]*DataSetReal["at"]


        ###
        angularFreqSpace = np.logspace(-2,4,100)
        DataSetFit = pd.DataFrame(data = angularFreqSpace, columns = ["Angular Frequency"])
        DataSetFit = self.workingWLFCrossModel(DataSetFit,params)
        print(DataSetFit)
        print(DataSetReal)



        fig, ax = plt.subplots(figsize=(14, 10))
        a2 = sns.lineplot(
            data=DataSetFit,  color = "black", markers = False, palette = hueOrder,#style = "Pressure [bar]",
            x="Angular Frequency", y="Complex viscosity", ax = ax, linewidth = 4
            );
        a1 = sns.scatterplot(
            data=DataSetReal,  style = "Temperature",  hue = "Temperature",
            x="Angular Frequency", y="Complex viscosity", palette = "flare", s = 100, edgecolor = "none", ax = ax
            );

        #plt.ylim(10,500000)
        plt.yscale("log")
        plt.xscale("log")

        #fig.set(xlim=(0.005,50000))
        #fig.set(yscale="log", xscale = "log"), "Complex Viscosity [Pa*s]" ,
        ax.set_xlabel( "Angular Frequency [rad/s]",fontsize = 24)
        ax.set_ylabel( "Complex Viscosity [Pa*s]", fontsize = 24)

        #for a1 in plt.axes.ravel():

        ax.tick_params(which="both", right=True, left= True,labelsize = 20)
        ax.tick_params(which="both", bottom = True, top=True, labelsize = 20)
        #ax.tick_params(which = "both",left=True, labelsize = 20)
        #ax.minorticks_on()
        plt.savefig("I:/My Drive/Research/OtherProjects/HiPressure/RawFigs/ComparisonBetweenModelsRealVisc.png", dpi = 800)

        plt.show()

     #graphs experimental vs fit data for WLF model
    def plotCompareData2(self,saveLoc):
        params = self.fitDataTotal()
        DataSetReal = self.SpecificDataSet
        print(params)
        DataSetReal["a_T"] = self.WLFModel(DataSetReal["Angular Frequency"],DataSetReal["Temperature"],self.TRef,params[0],params[1])
        DataSetReal["a_S"] = self.WLFModel(DataSetReal["Angular Frequency"],DataSetReal["Solubility"],self.SRef,params[2],params[3])
        #DataSetReal["a_P"] = self.WLFModel(DataSetReal["Angular Frequency"],DataSetReal["Solubility"],self.PRef,params[4],params[5])
        DataSetReal["a_total"] = DataSetReal["a_T"]* DataSetReal["a_S"]
        #print(DataSetReal["at"])
        DataSetReal["viscShift"] = DataSetReal["Complex viscosity"]/DataSetReal["a_total"]
        DataSetReal["freqShift"] = DataSetReal["Angular Frequency"]*DataSetReal["a_total"]


        ###
        angularFreqSpace = np.logspace(-2,4,100)
        DataSetFit = pd.DataFrame(data = angularFreqSpace, columns = ["Angular Frequency"])
        DataSetFit = self.workingWLFCrossModel(DataSetFit,params)
        print(DataSetFit)
        print(DataSetReal)



        fig, ax = plt.subplots(figsize=(14, 10))
        a2 = sns.lineplot(
            data=DataSetFit,  color = "black", markers = False,#style = "Pressure [bar]",
            x="Angular Frequency", y="Complex viscosity", ax = ax, linewidth = 4
            );
        a1 = sns.scatterplot(
            data=DataSetReal,  size = "Temperature",  hue = "Solubility",
            x="freqShift", y="viscShift", palette = "flare", s = 100, edgecolor = "none", ax = ax
            );

        #plt.ylim(10,500000)
        plt.yscale("log")
        plt.xscale("log")

        #fig.set(xlim=(0.005,50000))
        #fig.set(yscale="log", xscale = "log"), "Complex Viscosity [Pa*s]" ,
        ax.set_xlabel( "Angular Frequency [rad/s]",fontsize = 24)
        ax.set_ylabel( "Complex Viscosity [Pa*s]", fontsize = 24)

        #for a1 in plt.axes.ravel():

        ax.tick_params(which="both", right=True, left= True,labelsize = 20)
        ax.tick_params(which="both", bottom = True, top=True, labelsize = 20)
        #ax.tick_params(which = "both",left=True, labelsize = 20)
        #ax.minorticks_on()
        #plt.savefig("I:/My Drive/Research/OtherProjects/HiPressure/RawFigs/ComparisonBetweenModelsTanS.png", dpi = 800)

        plt.show()





####################################################################
#defining our fluid models.

    #WLF model with shift factor
    def WLFModel(self, x, T, TRef, C1, C2):

        return 10**(-C1*(T-TRef)/(C2+T-TRef))


    #total difference in model. Pulls a dataframe and fits parameters to it. For temperature only
    def WLFCrossModel(self,params):
        C1,C2,k,n,zeroS,infS = params
        Data = self.SpecificDataSet
        TRef = self.TRef

        ActualViscosity = Data["Complex viscosity"]
        Temperature = Data["Temperature"]
        x =  Data["Angular Frequency"]
        #WLFModel
        at = 10**(-C1*(Temperature-TRef)/(C2+Temperature-TRef))
        FunctionOut = ((zeroS-infS)/(1+(k*x*at)**n)+infS)*at
        return np.sum(np.abs(ActualViscosity - FunctionOut))

    #total difference in model. Pulls a dataframe and fits parameters to it. Fits solubility and temperature and pressure
    def WLFCrossModelPressure(self,params):
        C1,C2, C1S, C2S, k,n,zeroS,infS = params

        Data = self.SpecificDataSet
        TRef = self.TRef
        PRef = self.PRef

        ActualViscosity = Data["Complex viscosity"]
        Temperature = Data["Temperature"]
        Solubility = Data["Solubility"]

        Pressure = Data["Pressure"]
        x =  Data["Angular Frequency"]
        #WLFModel
        at_T = 10**(-C1*(Temperature-TRef)/(C2+Temperature-TRef))
        #at_P = 10**(-C1*(Pressure-self.PRef)/(C2+Pressure-self.PRef))

        at_S = 10**(-C1S*(Solubility-self.SRef)/(C2S+Solubility-self.SRef))
        at_total = at_T*at_S
        #print(at_S)
        FunctionOut = ((zeroS-infS)/(1+(k*x*at_total)**n)+infS)*at_total
        logFxn = np.log(FunctionOut)
        logData = np.log(ActualViscosity)


        residual = np.sum(np.abs(logFxn-logData))
        if residual < self.mini:
            self.mini = residual
            #print(residual)
        return residual

    #model for use and not for fitting. Good for getting outputs:
    #outputs np array of viscosity
    def workingWLFCrossModel(self,df,params):
        TRef = self.TRef
        C1,C2,C1s,c2s,k,n,zeroS,infS = params

        #df["at"] = 10**(-C1*(df["Temperature"]-TRef)/(C2+df["Temperature"]-TRef))
        df["Complex viscosity"]= ((zeroS-infS)/(1+(k*df["Angular Frequency"])**n)+infS)
        return df

    #includes data for T,P
    def workingWLFCrossModel2(self,df,params):
        TRef = self.TRef
        C1,C2,C1s,C2s,k,n,zeroS,infS = params

        #df["at"] = 10**(-C1*(df["Temperature"]-TRef)/(C2+df["Temperature"]-TRef))
        df["Complex viscosity"]= ((zeroS-infS)/(1+(k*df["Angular Frequency"])**n)+infS)
        return df

    def workingWLFCrossModel3(self,x,params):
        TRef = self.TRef
        C1,C2,C1s,C2s,k,n,zeroS,infS = params

        #df["at"] = 10**(-C1*(df["Temperature"]-TRef)/(C2+df["Temperature"]-TRef))
        output = ((zeroS-infS)/(1+(k*x)**n)+infS)
        return output


####################################################################
#user functions
    #fits a single set of temperature data
    def fitData(self):
        return self.fitWLFCrossModel(self.ReadData)

    #fits a  set of temperature and soluibility data

    def fitDataTotal(self):

        return self.fitWLFCrossModelPressure(self.ReadData)
    #Returns a shift factor for T

    def runWLFModelT(self, x, T):
        params = self.mostRecentParams
        return self.WLFModel(x,T,self.TRef,params[0],params[1])

    #Returns a shift factor for S
    def runWLFModelS(self, x, S):
        params = self.mostRecentParams
        return self.WLFModel(x,S,self.SRef,params[2],params[3])

    def CrossModelOut(self,x):
        params = self.mostRecentParams
        return self.workingWLFCrossModel3(x,params)

    #Returns solu data given T, P (C, Bar), returns in g/kg
    def evaluateSolubility(self,T,P):
        return self.SolubilityModel.ev(P,T)


##################################################################




#sixTBar = fitCrossAndWLF("./CombinedDataPS_p2.xlsx")
#params = sixTBar.fitDataTotal()
#sixTBar.plotCompareData2("loc")
