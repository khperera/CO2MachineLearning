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
#eta should be on order of 1?
#eta should be provided as eta*etaMax to scale eta
#eta =  1
#or eta is the exponent of viscosity
etaMax = 20
etaexp = dde.Variable(0.1, dtype=tf.float64)
#eta = 46
rho = 1000
torqueOut = 41887.9
expectedEta = 1000


eta = 10**etaexp
#########################
#some properties of system
#radius in meters
R = 1 #5 cm
#height in meteres
H = 0.05 #1 mm

AngularVelocity = 1

def slnfuc(x):
    return (2*3.14*AngularVelocity*expectedEta*(x)**3/3/H)


scaleFactor = H/(2*3.1415*expectedEta*AngularVelocity*R**3)
#########################
#make domain
#our domain is pseudo steady state, time invariant. Axisymmetric wrt theta
#geom = dde.geometry.Rectangle([0, 0], [R, H])

def solution1(x):

    return (2*3.14*expectedEta*AngularVelocity*x**3/3/H)


#define our sysyem?
def rheometerPlate(x,y):
    Tau =  y[:, 0:1]
    r = x[:,0:1]



    Tau_r =  dde.grad.jacobian(y,x,i=0)

    #eqn = ()
    return (Tau_r*H/(2*3.1415*(eta)*AngularVelocity*R**3)-(x/R)**2/R)
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)


#space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
geom = dde.geometry.Interval(0, R)


def boundary(_, on_initial):
    return on_initial

#ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)


def inside(x, on_boundary):
    return on_boundary and np.equal(x[0], 0)
def outside(x, on_boundary):
    return on_boundary and np.equal(x[0],R)



#inside, torqu needs to be  =0
bc = dde.DirichletBC(geom, lambda x: 0, inside)
bc2 = dde.DirichletBC(geom, lambda x: torqueOut, outside)
bc3 = dde.NeumannBC(geom, lambda x: 0, inside)
#swithc to neumann
#center, velocity is 0
#c24 = dde.NeumannBC(space_domain, lambda x: 0, Inside)


#bottom plate, velocity moves with angfular velocity
#bc3 = dde.DirichletBC(space_domain, boundary_condition, lambda x: np.abs(x[1]-Ly_min)<1e-5)

#edge, slip



data = dde.data.PDE(
    geom,
    rheometerPlate,
    [bc,bc2],
    num_domain=5000,
    num_boundary=2,
    solution = slnfuc
    #auxiliary_var_function=ex_func2,
)

# Define your PDE and BCs here
# ...
dde.config.set_default_float("float64")

net = dde.nn.FNN([1] + [50] * 4 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: y/(H/(2*3.1415*eta*AngularVelocity*R**3)))
#net.apply_input_transform(lambda x, y: x*R)
model = dde.Model(data, net)

external_trainable_variables = [eta]
variable = dde.callbacks.VariableValue(
    etaexp, period=200, filename="variables1.dat"
)


#plot_boundary_conditions(data)
n = 20000
name = "viscosityModel-non-Dim-CENTER0-5layer-vsico1000-2"
# train adam
"""

"""

model.compile(
"adam", lr=0.005,external_trainable_variables=etaexp,loss_weights=[10, 1, 1]
)
losshistory, train_state = model.train(iterations=n, callbacks = [variable])



model.compile("L-BFGS",external_trainable_variables=etaexp,loss_weights=[10, 1, 1])
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-35,
gtol=1.0e-35,
maxiter=50000,
maxfun=50000,
maxls=50,
)






losshistory, train_state = model.train(iterations=50000,callbacks = [variable])


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
