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

##viscosity initial guess
#etaMax = 20
#etaexp = dde.Variable(0.1, dtype=tf.float64)
#eta = 46


#########################
#some properties of system
#radius in meters
R = 1 #5 cm
#height in meteres
H = 0.05 #1 mm

AngularVelocity = 1



#scaleFactor = H/(2*3.1415*expectedEta*AngularVelocity*R**3)
#########################
#make domain
#our domain is pseudo steady state, time invariant. Axisymmetric wrt theta
#geom = dde.geometry.Rectangle([0, 0], [R, H])


def diffrheometerPlate(x,y):
    dc_t = dde.grad.jacobian(y, x, j=1)
    dc_r = dde.grad.jacobian(y, x, j=0)
    dc_rr = dde.grad.hessian(y, x, i=0, j = 0)

    t = x[:, 1:]
    r = x[:, 0:1]



    #eqn = ()
    return (dc_t - (dc_r/r) - dc_rr )
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)




## domain info
#space_domain = dde.geometry.Rectangle([Lx_min, Ly_min], [Lx_max, Ly_max])
geom = dde.geometry.Interval(0.0001, R)
timedomain = dde.geometry.TimeDomain(0, 2)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)















def boundary(_, on_initial):
    return on_initial

#ic1 = dde.icbc.IC(geom, lambda X: 1, boundary, component=0)


def inside(x, on_boundary):
    return on_boundary and np.equal(x[0], 0)
def outside(x, on_boundary):
    return on_boundary and np.equal(x[0],R)



#inside, torqu needs to be  =0
bc = dde.DirichletBC(geomtime, lambda x: 1, outside)
#bc2 = dde.DirichletBC(geom, lambda x: torqueOut, outside)
#bc3 = dde.NeumannBC(geom, lambda x: 0, inside)
ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)
#swithc to neumann
#center, velocity is 0
#c24 = dde.NeumannBC(space_domain, lambda x: 0, Inside)


#bottom plate, velocity moves with angfular velocity
#bc3 = dde.DirichletBC(space_domain, boundary_condition, lambda x: np.abs(x[1]-Ly_min)<1e-5)

#edge, slip



data = dde.data.TimePDE(
    geomtime,
    diffrheometerPlate,
    [bc,ic],
    num_domain=8000,
    num_boundary=100,
    num_initial = 100
    #solution = slnfuc
    #auxiliary_var_function=ex_func2,
)





# Define your PDE and BCs here
# ...
dde.config.set_default_float("float64")

net = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot uniform")
#net.apply_output_transform(lambda x, y: y/(H/(2*3.1415*eta*AngularVelocity*R**3)))
#net.apply_input_transform(lambda x, y: x*R)
model = dde.Model(data, net)

#external_trainable_variables = [eta]
#variable = dde.callbacks.VariableValue(
#    etaexp, period=200, filename="variables1.dat"
#)


#plot_boundary_conditions(data)
n = 25000
name = "viscosityModel-non-Dim-CENTER0-5layer-vsico1000-2"
# train adam
"""

"""

model.compile(
"adam", lr=0.01,loss_weights=[1, 10, 1]
)
losshistory, train_state = model.train(iterations=n)



model.compile("L-BFGS",loss_weights=[1, 10, 1])
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-35,
gtol=1.0e-35,
maxiter=100000,
maxfun=100000,
maxls=50,
)






losshistory, train_state = model.train(iterations=50000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
