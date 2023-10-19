import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ipywidgets as widgets
from IPython.display import display

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
