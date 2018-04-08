import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops


def _variable_with_weight_decay(name, shape, wd=1e-3):
    with tf.device("/cpu:0"): # store all weights in CPU to optimize weights sharing among GPUs
        var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def max_pool3d(input_, k, name='max_pool3d'):
  return tf.nn.max_pool3d(input_, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        w = _variable_with_weight_decay('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_dim])
        
        return tf.nn.bias_add(conv, b)


def cross_conv2d(input_, kernel, d_h=1, d_w=1, padding='SAME', name="cross_conv2d"):
    with tf.variable_scope(name):
        output_dim = kernel.get_shape()[4] if tf.flags.FLAGS.version == 1 else kernel.get_shape()[3]
        batch_size = input_.get_shape().as_list()[0]
        b = _variable_with_weight_decay('b', [output_dim])
       
        output = []
        input_list  = tf.unstack(input_)
        kernel_list = tf.unstack(kernel)
        for i in range(batch_size):
            if tf.flags.FLAGS.version == 1:
                conv = tf.nn.conv2d(tf.expand_dims(input_list[i],0), kernel_list[i], strides=[1, d_h, d_w, 1], padding=padding)
            elif tf.flags.FLAGS.version in [2,3,4]:
                conv = tf.nn.depthwise_conv2d(tf.expand_dims(input_list[i],0), kernel_list[i], strides=[1, d_h, d_w, 1], padding=padding)
            conv = tf.nn.bias_add(conv, b)
            output.append(conv)

    return tf.concat(output, 0)


def conv3d(input_, output_dim, k_t=3, k_h=3, k_w=3, d_t=1, d_h=1, d_w=1, padding='SAME', name="conv3d"):
    with tf.variable_scope(name):
        w = _variable_with_weight_decay('w', [k_t, k_h, k_w, input_.get_shape()[-1], output_dim])
        conv = tf.nn.conv3d(input_, w, strides=[1, d_t, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_dim])
        
        return tf.nn.bias_add(conv, b)


def relu(x):
    return tf.nn.relu(x)


def fc(input_, output_dim, name='fc'):
    with tf.variable_scope(name):
        w = _variable_with_weight_decay('w', [input_.get_shape()[-1], output_dim])
        b = _variable_with_weight_decay('b', [output_dim])
        
        return tf.matmul(input_, w) + b


def deconv2d(input_, output_shape, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = _variable_with_weight_decay('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_shape[-1]])

        return tf.nn.bias_add(deconv, b)


def deconv3d(input_, output_shape, k_t=3, k_h=3, k_w=3, d_t=1, d_h=1, d_w=1, padding='SAME', name="deconv3d"):
    with tf.variable_scope(name):
        # filter : [depth, height, width, output_channels, in_channels]
        w = _variable_with_weight_decay('w', [k_t, k_h, k_h, output_shape[-1], input_.get_shape()[-1]])
        deconv = tf.nn.conv3d_transpose(input_, w, output_shape=output_shape, strides=[1, d_t, d_h, d_w, 1], padding=padding)
        b = _variable_with_weight_decay('b', [output_shape[-1]])

        return tf.nn.bias_add(deconv, b)        


def shape2d(a):
    """
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))


def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret


def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.
    Args:
        x (tf.Tensor): a NHWC tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.
    Returns:
        tf.Tensor: a NHWC tensor.
    """
    shape = shape2d(shape)

    # a faster implementation for this special case
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
        return UnPooling2x2ZeroFilled(x)

    input_shape = tf.shape(x)
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.constant(mat, name='unpool_mat')
    elif isinstance(unpool_mat, np.ndarray):
        unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)

    # perform a tensor-matrix kronecker product
    fx = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1])
    fx = tf.expand_dims(fx, -1)       # (bchw)x1
    mat = tf.expand_dims(tf.reshape(unpool_mat, [-1]), 0)  # 1x(shxsw)
    prod = tf.matmul(fx, mat)  # (bchw) x(shxsw)
    prod = tf.reshape(prod, tf.stack(
        [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]]))
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, tf.stack(
        [-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3]]))
    return prod      
