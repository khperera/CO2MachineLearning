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
import seaborn as sns
import matplotlib.pyplot as plt
from CO2Fxn import fitCrossAndWLF
from CO2V10inverse import CO2Sim


files = ["20-150","80-130","40-150", "100-130","80-150"]
#files = ["80-130"]

for file in files:
    filename = "./TrainingData/"+file+".csv"
    #thing = pd.read_csv(filename)

    sim1 = CO2Sim(file,filename)
    #sim1.loadModel('./testcase1-27681.ckpt')
    #sim1.runAdam(10000,[1,1,1,100], 0.005, False)
    sim1.runAdam(10000,[1,1,1,100], 0.005, False)
    sim1.runBFGS(60000,[10,1,1,10], False)
    sim1.saveModel(file)
    #sim1.savePlot()
    del sim1
