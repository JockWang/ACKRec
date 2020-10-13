from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.test = []

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, length, placeholders, tag, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support_'+tag]
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.tag = tag
        self.length = length

        # helper variable for sparse dropout
        # self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name+ '_' + self.tag + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
                # self.vars['bias_'+str(i)] = zeros([output_dim,], name='bias_' + str(i))
                self.vars['bias_' + str(i)] = tf.zeros(shape=(self.length, 1), name='bias_' + str(i))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        supports = list()
        for i in range(len(self.support)):
            if self.name == 'first'+self.tag: #这里注释了
                x = inputs
            else:
                x = inputs[i]
            # x = inputs #做成concat需要修改三个地方，修改这里的输入，add输出，移除attention

        # dropout
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        #     support = tf.matmul(self.support[i], x)
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)])
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup)
            # self.test.append(self.vars['bias_' + str(i)])
            support = support + self.vars['bias_' + str(i)]
            supports.append(self.act(support))
        # output = tf.add_n(supports) #这里解除注释了
        output = supports #这里注释了
        # bias
        # return output
        return self.act(output)

class RatLayer():
    def __init__(self, user, item, act=tf.nn.relu):
        self.user = user
        self.item = item
        self.act = act

    def __call__(self):
        rate_matrix = tf.matmul(self.user,tf.transpose(self.item))
        return self.act(rate_matrix)


class RateLayer():
    def __init__(self, user, item, user_dim, item_dim, ac=tf.nn.relu):
        self.user = user
        self.item = item
        self.name = 'RateLayer'
        self.ac = ac
        self.vars = {}
        with tf.name_scope(self.name + '_vars'):
            self.vars['user_latent'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.latent_dim), user_dim],
                                                                       stddev=1.0), name='user_latent_matrix')
            self.vars['item_latent'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.latent_dim), item_dim],
                                                                       stddev=1.0), name='item_latent_matrix')
            self.vars['user_specific'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.output_dim), item_dim],
                                                                         stddev=0.1), name='user_specific')
            self.vars['item_specific'] = tf.Variable(tf.truncated_normal(shape=[int(FLAGS.output_dim), user_dim],
                                                                         stddev=0.1), name='item_specific')
            self.vars['user_bias'] = tf.zeros(shape=[user_dim,1],name='user_bias')
            self.vars['item_bias'] = tf.zeros(shape=[item_dim,1], name='item_bias')
            self.vars['alpha1'] = tf.Variable(initial_value=1.0, name='alpha1')
            self.vars['alpha2'] = tf.Variable(initial_value=1.0, name='alpha2')

    def __call__(self):
        rate_matrix1 = tf.matmul(tf.transpose(self.vars['user_latent']),self.vars['item_latent'])
        u_matrix = self.vars['alpha1']*(tf.matmul(self.user, self.vars['user_specific'])+self.vars['user_bias'])
        i_matrix = self.vars['alpha2']*(tf.transpose(tf.matmul(self.item,
                                                               self.vars['item_specific'])+self.vars['item_bias']))
        rate_matrix2 = rate_matrix1+u_matrix+i_matrix
        return rate_matrix2

class SimpleAttLayer():
    def __init__(self, attention_size, tag, time_major=False):
        self.attention_size = attention_size
        self.time_major = time_major
        self.tag = tag
        self.vars = {}

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if self.time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters

        with tf.variable_scope('v_'+self.tag):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            w_omega = tf.get_variable(initializer=tf.random_normal([hidden_size, self.attention_size],
                                                                   stddev=0.1), name='w_omega')
            self.vars['w_omega'] = w_omega
            b_omega = tf.get_variable(initializer=tf.random_normal([self.attention_size], stddev=0.1), name='b_omega')
            self.vars['b_omega'] = b_omega
            u_omega = tf.get_variable(initializer=tf.random_normal([self.attention_size], stddev=0.1), name='u_omega')
            self.vars['u_omega'] = u_omega
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape
        self.alphas = vu

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape

        output = tf.reduce_sum(inputs*tf.expand_dims(alphas, -1), 0)

        return output