#simple rheometer flow

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax"""
#import deepxde as dde
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import special as spe
#from deepxde.backend import tfhelper
#from deepxde.boundarycondition import BC
print(tf.__version__)

#constants
#r = [1,2,3,4,5,6,7,8,9]
#t = [1,2,3,4,5]
D = 1e-4
R = 5
T = 100000








# Define tensors for a_n and r (assuming 1D tensors for demonstration)
#a_n = tf.constant([0.1, 0.2, 0.3, 0.4,4,5,6], dtype=tf.float32)
numberofZeroes = 500
a_n = tf.constant(spe.jn_zeros(0, numberofZeroes), dtype=tf.float32)
r1 = np.arange(0.001,R,0.05)
t1 = np.arange(0,T,10000)

print(r1)
r = tf.constant(r1, dtype=tf.float32)
t = tf.constant(t1, dtype=tf.float32)
# Reshape to enable broadcasting
a_n = tf.reshape(a_n, [-1, 1,1])  # Shape becomes [M, 1]
r = tf.reshape(r, [1, -1,1])      # Shape becomes [1, N]
t =tf.reshape(t, [1, 1,-1])

# Compute bessel_j0(a_n * r / 5)


bessel_result = tf.math.special.bessel_j0(a_n * r / R)/(tf.math.special.bessel_j1(a_n)*a_n)*tf.exp(-a_n**2*D*t/R**2)




# Sum along the axis corresponding to a_n (axis=0)
result_sum = tf.reduce_sum(bessel_result, axis=0)


# Create a MultiIndex using the r1 and t1 arrays
index = pd.MultiIndex.from_product([r1, t1], names=['r', 't'])
# Flatten the bessel_result_sum array to match the MultiIndex
bessel_result_1d = tf.reshape(result_sum, [-1])
# Convert the tensor to a NumPy array
bessel_result_1d_np = 1-2*np.abs(bessel_result_1d.numpy())
# Create the DataFrame
df = pd.DataFrame({'Bessel_Sum': bessel_result_1d_np}, index= index)
print(df)


tf.print(result_sum, output_stream=sys.stdout)

fig1 = sns.scatterplot(data = df, x = "r", y = "Bessel_Sum", hue = "t")
plt.show()
"""
## zeros of bessel fxn
#bessel fxn
numberofZeroes = 5
an = spe.jn_zeros(0, numberofZeroes)
print(an)
an = tf.reshape(an, [1, 5])







C = 1 - 2*tf.reduce_sum(tf.math.special.bessel_j0(an*r/5)/(an*tf.math.special.bessel_j1(an))*tf.exp(-an**2*D*t/5**2),0)




tf.print(C, output_stream=sys.stdout)
"""
