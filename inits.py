import numpy as np
import tensorflow as tf

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    # initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    # initial = tf.random_uniform(shape, stddev=0.1)
    # return tf.Variable(tf.random_normal(shape, stddev=1.0))
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)