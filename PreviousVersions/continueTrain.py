import tensorflow as tf
from tensorflow import keras
import deepxde as dde

# Re-define your model architecture here (must be the same as the saved model)
net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")






#simple rheometer flow

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
import deepxde as dde
import numpy as np
import matplotlib as plt
import numpy as np
import tensorflow as tf
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC



#Import BC at some point

#viscosity initial guess
eta =  dde.Variable(1.0, dtype=tf.float64)

#########################
#some properties of system
#radius in meters
R = 1 #5 cm
#height in meteres
H = 0.01 #1 mm
Lx_min = 0.0001
Lx_max = R
Ly_min = 0.0001
Ly_max = H
AngularVelocity = 1

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
    Vdtheta =  y[:, 0:1]
    r = x[:,0:1]
    z = x[:,1:2]


    mult_Vtheta_r = Vdtheta*r
    print(mult_Vtheta_r)
    dVtheta_r = dde.grad.jacobian(mult_Vtheta_r, x, i=0)

    Vtheta_rdiv =  dVtheta_r/(r)
    dVtheta_r_r = dde.grad.jacobian(Vtheta_rdiv, x, i=0)
    dVtheta_zz = dde.grad.hessian(Vdtheta,x,i=1, j = 1)



     # calculate dV/dx
    #V_x = tf.gradients(V, x1)[0]

    # calculate d/dx(1/x*dV/dx)
    # term = tf.gradients((1 / x1) * V_x, x1)[0]
    #we have indep variables t, theta and z, we have dep variabels theta. We have first order and second order things

    #eqn = ()
    return eta*(dVtheta_r_r+dVtheta_zz)
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)
def rheometerPlate2(x,y):
    Vdtheta =  y[:, 0:1]
    r = x[:,0:1]
    z = x[:,1:2]


    mult_Vtheta_r = Vdtheta*r
    print(mult_Vtheta_r)
    dVtheta_r = dde.grad.jacobian(y, x, i=0)
    dVtheta_r_r = dde.grad.hessian(Vdtheta,x,i=0, j = 0)
    #Vtheta_rdiv =  dVtheta_r/(r)
    #dVtheta_r_r = dde.grad.jacobian(y, x, i=0)
    dVtheta_zz = dde.grad.hessian(Vdtheta,x,i=1, j = 1)



     # calculate dV/dx
    #V_x = tf.gradients(V, x1)[0]

    # calculate d/dx(1/x*dV/dx)
    # term = tf.gradients((1 / x1) * V_x, x1)[0]
    #we have indep variables t, theta and z, we have dep variabels theta. We have first order and second order things

    #eqn = ()
    return eta*(dVtheta_r_r+dVtheta_zz)
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)



space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])

def boundary(_, on_initial):
    return on_initial

#ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)

def boundary(_, on_initial):
    return on_initial

#ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)


def topplate(x, on_boundary):
    return on_boundary and np.isclose(x[1], Ly_max)

def funcBoundary(x):
    return AngularVelocity*x[:,0:1] # for example

def boundary2(x, on_boundary):
    return on_boundary and np.isclose(x[0],Lx_min)
def boundary3(x, on_boundary):
    return on_boundary and np.isclose(x[1], Ly_min)
def boundary4(x, on_boundary):
    return on_boundary and np.isclose(x[0], Lx_max)


def boundary_condition(x):
    return m * x[0] + b  # assuming x[1] corresponds to 'y'


#Top plate, needs to be velocity =0
bc = dde.DirichletBC(space_domain, lambda x: 0, topplate)

#center, velocity is 0
#bc2 = dde.DirichletBC(space_domain, lambda x: 1, boundary2)
#swithc to neumann
bc2 = dde.NeumannBC(space_domain, lambda x: 0, boundary2)


#bottom plate, velocity moves with angfular velocity
#bc3 = dde.DirichletBC(space_domain, boundary_condition, lambda x: np.abs(x[1]-Ly_min)<1e-5)
bc3 = dde.DirichletBC(space_domain, funcBoundary, boundary3)

#edge, slip
bc4 = dde.NeumannBC(space_domain, lambda x: 0, boundary4)



data = dde.data.PDE(
    space_domain,
    rheometerPlate2,
    [bc, bc2, bc3, bc4],
    num_domain=2000,
    num_boundary=500
    #auxiliary_var_function=ex_func2,
)






























# Define your PDE and BCs here
# ...
dde.config.set_default_float("float64")
# Compile the model
model = dde.Model(data, net)

model.compile("adam", lr=0.01)

# Load the weights from the saved model
model.restore('./viscosityModel-accurateDimensions-150007.ckpt')

# Continue training
losshistory, train_state = model.train(iterations=10000)
model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e7, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e6}}

model.compile("L-BFGS")
losshistory, train_state = model.train(iterations=100000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

model.compile("adam", lr=0.001)
# Continue training
losshistory, train_state = model.train(iterations=100000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save("./viscosityModel-p3")
