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
eta =  dde.Variable(0.01, dtype=tf.float64)
rho = dde.Variable(1000, dtype=tf.float64)





#########################
#some properties of system
#radius in meters
R = 0.5 #5 cm
#height in meteres
H = 0.001 #1 mm
Lx_min = 0.0001
Lx_max = 1
Ly_min = 0.0001
Ly_max = 1
AngularVelocity = 10

def initial_guess(x):
    r = x[:, 0:1]
    z = x[:, 1:2]
    # Linear interpolation between velocity at z = 0 (increasing with r) and z = H (0 velocity)
    V = AngularVelocity*r * (1 - z/H)
    return V

#########################
#make domain
#our domain is pseudo steady state, time invariant. Axisymmetric wrt theta
#geom = dde.geometry.Rectangle([0, 0], [R, H])


#define our sysyem?
#x represents r,z, y represents theta
def rheometerPlate(x,y):
    Vt =  y[:, 0:1]
    r = x[:,0:1]
    z = x[:,1:2]
    nonDimensionalizer = eta/AngularVelocity/rho



    Vt_r = dde.grad.jacobian(y, x, i=0)
    Vt_rr =  dde.grad.hessian(y,x,i=0, j = 0)
    Vt_zz =  dde.grad.hessian(y,x,i=1, j = 1)

    #eqn = ()
    return nonDimensionalizer*(Vt_rr/(R**2) + 1/r*Vt_r/(R**2)+Vt_zz/(H**2)-Vt/r/(R))
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)
def rheometerPlateCart(x,y):
    Vx =  y[:, 0:1]
    y = x[:,0:1]
    z = x[:,1:2]
    nonDimensionalizer = eta/AngularVelocity/rho


    dVtheta_r_r = dde.grad.hessian(Vx,x,i=0, j = 0)
    #Vtheta_rdiv =  dVtheta_r/(r)
    #dVtheta_r_r = dde.grad.jacobian(y, x, i=0)
    dVtheta_zz = dde.grad.hessian(Vx,x,i=1, j = 1)



     # calculate dV/dx
    #V_x = tf.gradients(V, x1)[0]

    # calculate d/dx(1/x*dV/dx)
    # term = tf.gradients((1 / x1) * V_x, x1)[0]
    #we have indep variables t, theta and z, we have dep variabels theta. We have first order and second order things

    #eqn = ()
    return nonDimensionalizer*(dVtheta_r_r/R/R+dVtheta_zz/H/H)
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)

def rheometerPlate3(x,y):
    Vdtheta =  y[:, 0:1]
    r = x[:,0:1]#dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    z = x[:,1:2]


    nonDimensionalizer = eta/AngularVelocity/rho
    dVtheta_r = dde.grad.jacobian(y, x, i=0)
    #dVtheta_r_r = dde.grad.hessian(Vdtheta,x,i=0, j = 0)
    #Vtheta_rdiv =  dVtheta_r/(r)
    #dVtheta_r_r = dde.grad.jacobian(y, x, i=0)
    dVtheta_zz = dde.grad.hessian(Vdtheta,x,i=1, j = 1)



     # calculate dV/dx
    #V_x = tf.gradients(V, x1)[0]

    # calculate d/dx(1/x*dV/dx)
    # term = tf.gradients((1 / x1) * V_x, x1)[0]
    #we have indep variables t, theta and z, we have dep variabels theta. We have first order and second order things

    #eqn = ()
    return nonDimensionalizer*(dVtheta_r/r/R/R+dVtheta_zz/H/H-Vdtheta/r/r/R/R)



space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])

def boundary(_, on_initial):
    return on_initial

#ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)


def topplate(x, on_boundary):
    return on_boundary and np.isclose(x[1], Ly_max)

def funcBoundary(x):
    return AngularVelocity*x[:,0:1] # for example

def funcBoundary2(x):
    return AngularVelocity*(1-x[:,1:2]) # for example

def Inside(x, on_boundary):
    return on_boundary and np.isclose(x[0],Lx_min)
def BottomPlate(x, on_boundary):
    return on_boundary and np.isclose(x[1], Ly_min)
def Outside(x, on_boundary):
    return on_boundary and np.isclose(x[0], Lx_max)


def boundary_condition(x):
    return m * x[0] + b  # assuming x[1] corresponds to 'y'


#Top plate, needs to be velocity =0
bc = dde.DirichletBC(space_domain, lambda x: 0, topplate)

#center, velocity is 0
bc2 = dde.DirichletBC(space_domain, lambda x: 0, Inside)
#swithc to neumann
#c24 = dde.NeumannBC(space_domain, lambda x: 0, Inside)


#bottom plate, velocity moves with angfular velocity
#bc3 = dde.DirichletBC(space_domain, boundary_condition, lambda x: np.abs(x[1]-Ly_min)<1e-5)
bc3 = dde.DirichletBC(space_domain, funcBoundary, BottomPlate)

#edge, slip
bc4 = dde.DirichletBC(space_domain, funcBoundary2, Outside)



data = dde.data.PDE(
    space_domain,
    rheometerPlateCart,
    [bc, bc2, bc3, bc4],
    num_domain=5000,
    num_boundary=1000
    #auxiliary_var_function=ex_func2,
)

# Define your PDE and BCs here
# ...
dde.config.set_default_float("float64")

net = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)

external_trainable_variables = []
#variable = dde.callbacks.VariableValue(
#    external_trainable_variables, period=200, filename="variables.dat"
#)


#plot_boundary_conditions(data)
n = 10000
name = "viscosityModel-non-Dim-CENTER0-5layer-vsico1000-2"
# train adam
model.compile(
    "adam", lr=0.01
)
losshistory, train_state = model.train(iterations=n)
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)
#model.save("./"+name)







#locationofmodel = "./"+name+"-"+str(n)+".ckpt"
# Load the weights from the saved model
#model.restore(locationofmodel)







# train lbfgs
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
#model.compile("L-BFGS")
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-30,
gtol=1.0e-35,
maxiter=50000,
maxfun=50000,
maxls=50,
)
model.compile("L-BFGS")





losshistory, train_state = model.train(iterations=100000)

model.compile(
    "adam", lr=0.001
)
losshistory, train_state = model.train(iterations=n)


dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-100,
gtol=1.0e-100,
maxiter=50000,
maxfun=50000,
maxls=50,
)
model.compile("L-BFGS")
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-100,
gtol=1.0e-100,
maxiter=50000,
maxfun=50000,
maxls=50,
)






losshistory, train_state = model.train(iterations=50000)


#dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save("./"+name+"-pt2")

model.compile(
    "adam", lr=0.001
)
losshistory, train_state = model.train(iterations=n)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# train adam
#model.compile(
#    "adam", lr=0.0001, external_trainable_variables=external_trainable_variables
#)
#losshistory, train_state = model.train(iterations=1000, callbacks=[variable])
#dde.saveplot(losshistory, train_state, issave=True, isplot=True)
