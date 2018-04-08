import tensorflow as tf

from ops import *


x  = tf.random_normal([16, 64, 64, 64])
gt = tf.random_normal([16, 64, 64, 3])

#x  = FixedUnPooling(x, [2,2])
y  = relu(deconv2d(x, output_shape=[16,64,64,3],
                        k_h=3,k_w=3,d_h=1,d_w=1,name='deconv1'))
print(y.get_shape().as_list())
#y  = FixedUnPooling(y, [2,2])

loss = tf.reduce_mean(tf.abs(y-gt))

opt = tf.train.AdamOptimizer(1e-4)

tf.get_variable_scope().reuse_variables()

vars_to_optimize = tf.trainable_variables()
grads = opt.compute_gradients(loss, var_list=vars_to_optimize)
