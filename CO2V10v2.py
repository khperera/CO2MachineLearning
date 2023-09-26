#simple rheometer flow

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
import deepxde as dde
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import special as spe
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC



dde.config.set_default_float("float64")
dde.config.real.set_float64()
#Import BC at some point

#viscosity initial guess
#eta should be on order of 1?
#eta should be provided as eta*etaMax to scale eta
#eta =  1


#########################
#some properties of system
#radius in meters
R = 1 #5 cm
#height in meteres
H = 0.05 #1 mm

AngularVelocity = 1

#or eta is the exponent of viscosity
etaMax = 20
#etaexp = dde.Variable(0.1, dtype=tf.float64)
#eta = 46
rho = 1000
torqueOut = 41887.9
expectedEta = 1000
D = 5e-4
zeroShear = 1
#eta = 10**etaexp
maxEta = 10
#######################
#setting up concentration profile of system
etaSample = 5


#bessel fxn
numberofZeroes = 5

a_n = tf.constant(spe.jn_zeros(0, numberofZeroes), dtype=tf.float64)
a_n = tf.reshape(a_n, [1, 1,-1])
#radial fxn of concentration v review

def concentrationSln(r,t,R,a_n,D):
    #r = tf.reshape(r1,[-1,-1,1])
    #t = tf.reshape(t1,[-1,-1,1])
    bessel_result = tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*D*t/R**2)
    result_sum = 1-2*tf.reduce_sum(bessel_result, axis=2)



    return result_sum



def concentrationToViscosity(concentration, zeroShear):
    return concentration + zeroShear

def calcEta(r,t,R,a_n,D,zeroShear):
    return concentrationToViscosity(concentrationSln(r,t,R,a_n,D), zeroShear)




scaleFactor = H/(2*3.1415*expectedEta*AngularVelocity*R**3)
#########################
#make domain
#our domain is pseudo steady state, time invariant. Axisymmetric wrt theta
#geom = dde.geometry.Rectangle([0, 0], [R, H])


#define our sysyem?
def rheometerPlate(x,y):
    Tau =  y[:, 0:1]
    x1 =tf.reshape(x, [-1,-1,1])

    r = x[:,0:1]
    t = x[:, 1:2]

    #print("Shape of tensor:", tf.shape(t))
    #r1 = x1[:,0:1,:]
    #t1 = x1[:, 1:,:]
    #print("Shape of tensor:", tf.shape(t1))

    Tau_r =  dde.grad.jacobian(y,x,i=0)


    #eta = concentrationToViscosity(concentration,zeroShear)
    eta = 5


    #etaSample = eta


    #eqn = ()
    return (Tau_r*H/(2*3.1415*(eta)*AngularVelocity*R**3)-(x/R)**2/R)
    #return (eta*dVtheta_zz)
    #return (eta*dVtheta_r_r)



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


##################################
#Boundary conditions

#inside, torqu needs to be  =0
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=0)
#bc = dde.icbc.DirichletBC(geomtime, lambda x: torqueOut, lambda _, on_initial: on_initial, component=0)
#bc2 = dde.icbc.DirichletBC(geomtime, lambda x: torqueOut, outside)
bc3 = dde.icbc.NeumannBC(geomtime, lambda x: 0, lambda _, on_initial: on_initial, component=0)




#ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)








data = dde.data.TimePDE(
    geomtime,
    rheometerPlate,
    [bc,bc3],
    num_domain=8000,
    num_boundary=100,
    num_initial = 100
    #solution = slnfuc
    #auxiliary_var_function=ex_func2,
)

# Define your PDE and BCs here
# ...
dde.config.set_default_float("float64")

net = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: y/(H/(2*3.1415*1*AngularVelocity*R**3)))
#net.apply_input_transform(lambda x, y: x*R)
model = dde.Model(data, net)

#external_trainable_variables = [eta]
#variable = dde.callbacks.VariableValue(
#    etaexp, period=200, filename="variables1.dat"
#)calcEta(x[:,0:1],x[:, 1:],R,a_n,D,zeroShear)


#plot_boundary_conditions(data)
n = 20000
name = "viscosityModel-non-Dim-CENTER0-5layer-vsico1000-2"
# train adam
"""

"""

model.compile(
"adam", lr=0.005,loss_weights=[10, 1, 1]
)
losshistory, train_state = model.train(iterations=n)



model.compile("L-BFGS",loss_weights=[10, 1, 1])
#model.train_step.optimizer_kwargs = {'options': {'maxfun': 1e5, 'ftol': 1e-20, 'gtol': 1e-20, 'eps': 1e-20, 'iprint': -1, 'maxiter': 1e5}}
dde.optimizers.config.set_LBFGS_options(
maxcor=100,
ftol=1.0e-35,
gtol=1.0e-35,
maxiter=50000,
maxfun=50000,
maxls=50,
)






losshistory, train_state = model.train(iterations=50000)


dde.saveplot(losshistory, train_state, issave=True, isplot=True)
