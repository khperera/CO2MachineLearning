#simple rheometer flow

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
import deepxde as dde
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC
dde.config.set_default_float("float64")
dde.config.real.set_float64()
#Import BC at some point

#viscosity initial guess
eta =  dde.Variable(1, dtype=tf.float64)
rho = dde.Variable(1000, dtype=tf.float64)





#########################
#some properties of system
#radius in meters
R = 10 #5 cm
#height in meteres
H = 1 #1 mm

AngularVelocity = 1


#########################
#make domain
#our domain is pseudo steady state, time invariant. Axisymmetric wrt theta
#geom = dde.geometry.Rectangle([0, 0], [R, H])

def solution1(x):

    return (2*3.14*eta*AngularVelocity*x**3/3/H)


#define our sysyem?
def rheometerPlate(x,y):
    Tau =  y[:, 0:1]
    r = x[:,0:1]



    Tau_r =  dde.grad.jacobian(y,x,i=0)

    #eqn = ()
    return (Tau_r-2*eta*AngularVelocity*3.1415*x**2/H)
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)


#space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
geom = dde.geometry.Interval(0, R)


def boundary(_, on_initial):
    return on_initial

#ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)


def inside(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)
def outside(x, on_boundary):
    return on_boundary and np.isclose(x,R)



#inside, torqu needs to be  =0
bc = dde.DirichletBC(geom, lambda x: 0, inside)

#swithc to neumann
#center, velocity is 0
#c24 = dde.NeumannBC(space_domain, lambda x: 0, Inside)


#bottom plate, velocity moves with angfular velocity
#bc3 = dde.DirichletBC(space_domain, boundary_condition, lambda x: np.abs(x[1]-Ly_min)<1e-5)

#edge, slip



data = dde.data.PDE(
    geom,
    rheometerPlate,
    [bc],
    num_domain=5000,
    num_boundary=500
    #auxiliary_var_function=ex_func2,
)

# Define your PDE and BCs here
# ...
dde.config.set_default_float("float64")

net = dde.nn.FNN([1] + [50] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

external_trainable_variables = []
#variable = dde.callbacks.VariableValue(
#    external_trainable_variables, period=200, filename="variables.dat"
#)


#plot_boundary_conditions(data)
n = 100000
name = "viscosityModel-non-Dim-CENTER0-5layer-vsico1000-2"
# train adam
"""
model.compile("L-BFGS")
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-20,
gtol=1.0e-20,
maxiter=50000,
maxfun=50000,
maxls=50,
)






losshistory, train_state = model.train(iterations=50000)
"""

model.compile(
"adam", lr=0.01
)
losshistory, train_state = model.train(iterations=n)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
