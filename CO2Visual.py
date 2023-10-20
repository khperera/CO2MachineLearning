import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
#import ipywidgets as widgets
#from IPython.display import display
from CO2Fxn import fitCrossAndWLF

class dataAnalysis:

    def __init__(self):
        self.fileLocations = []
        self.LiteratureDataLoc = "./TrainingData/MiscFiles/DataFromLiterature.xlsx"
        self.LiteratureData = ""
        self.DConstants = []
        self.collectedData = ""
        self.SolubilityViscModel = fitCrossAndWLF("./CombinedDataPS_p2.xlsx")



    def readFiles(self, folderLocation):
        self.fileLocations = os.listdir(folderLocation)
        print(self.fileLocations)
        for file in self.fileLocations:
            if ".swp" in file:
                continue
            #print(file)
            substrings = file.split("-")
            Pressure = substrings[0]
            Temperature = substrings[1].replace("variables1.dat","")
            dataf1 = pd.read_table(folderLocation+file, sep = " ", names = ["Epoch","Value"])
            dataf1["Value1"] =  dataf1["Value"].str.replace("[","").str.replace("]","").astype(float)
            #dataf1["Value1"] =  dataf1["Value"].astype(float)

            dataf1["DConstant"] = 10**(np.tanh(dataf1["Value1"])-9)
            #print(dataf1)
            dconst = dataf1.iloc[-1]["DConstant"]
            tupleOfData = (Pressure,Temperature,dconst)
            self.DConstants.append(tupleOfData)
            #print(dconst)
        self.collectedData = pd.DataFrame(data = self.DConstants, columns = ["Pressure", "Temperature", "D"])
        self.collectedData["Temperature (K)"] = self.collectedData["Temperature"].astype(float)+273.13
        self.collectedData["Average pressure (MPa)"] = self.collectedData["Pressure"].astype(float)/10
        #self.collectedData["Average solubility (g-gas kg-polym)"] = self.SolubilityViscModel.evaluateSolubility(100,20.6)
        self.collectedData["From"] = "ML Calculations"
        self.collectedData["Average solubility (g-gas kg-polym)"] = self.SolubilityViscModel.evaluateSolubility(self.collectedData["Temperature"].astype(float),self.collectedData["Pressure"].astype(float))
        #self.collectedData["Pressure (MPa)"] = self.collectedData["Temperature"]


        print(self.collectedData)

    def readLitData(self):
        self.LiteratureData = pd.read_excel(self.LiteratureDataLoc)
        #print(self.LiteratureData )

    def plotAllData(self):
        totalData = pd.concat([self.collectedData,self.LiteratureData])
        print(totalData)
        fig, ax = plt.subplots(figsize=(14, 10))
        a1 = sns.scatterplot(
            data=totalData,  style = "From",  x = "Temperature (K)",
            hue="Average solubility (g-gas kg-polym)", y="D", palette = "flare", s = 100, edgecolor = "none", ax = ax
            );
        plt.yscale("log")
        plt.show()



    def someRandomFxn(self):

        # Load your data
        train_data = np.loadtxt('train.dat')
        test_data = np.loadtxt('test.dat')

        def plot_3d(elev=0., azim=0.):
            fig = plt.figure(figsize=(10, 8))
            # Subplot for train data
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2])
            ax1.view_init(elev=elev, azim=azim)
            ax1.set_title('Train Data')
            # Subplot for test data
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2])
            ax2.view_init(elev=elev, azim=azim)
            ax2.set_title('Test Data')
            plt.show()

        # Create widgets
        elev_slider = widgets.FloatSlider(min=-90., max=90., step=5., value=0.)
        azim_slider = widgets.FloatSlider(min=-180., max=180., step=5., value=0.)

        # Create interactive plot
        widgets.interact(plot_3d, elev=elev_slider, azim=azim_slider)

data1 = dataAnalysis()

data1.readFiles("./variableTrained/")
data1.readLitData()
data1.plotAllData()
