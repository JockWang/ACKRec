import tensorflow as tf
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.mrr = 0
        self.hrat1 = 0
        self.hrat5 = 0
        self.hrat10 = 0
        self.hrat20 = 0
        self.ndcg1 = 0
        self.ndcg5 = 0
        self.ndcg10 = 0
        self.ndcg20 = 0
        self.test = None
        self.alphas = None
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for i in range(len(self.layers)):
            hidden = self.layers[i](self.activations[-1])
            if i == 3:
                self.test = hidden
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self.alphas = self.layers[3].alphas

        # Store model variables for easy access
        # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        # self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._hrat()
        self._ndcg()
        self._accuracy()
        self._mrr()

        self.opt_op = self.optimizer.minimize(self.loss)


    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def _hrat(self):
        raise NotImplementedError

    def _mrr(self):
        raise NotImplementedError

    def _ndcg(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    def __init__(self, placeholders, input_dim, num_support, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.rating = placeholders['rating']
        self.negative = placeholders['negative']
        self.length = int(self.rating.shape[0])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(4):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # RMSE
        self.loss += rmse_loss(rating=self.rating,rate=self.outputs, length=self.length)

    def _accuracy(self):
        self.accuracy = auc(self.outputs, self.negative, length=self.length)

    def _ndcg(self):
        self.ndcg1 = ndcg(self.outputs, self.negative, length=self.length, k=1)
        self.ndcg5 = ndcg(self.outputs, self.negative, length=self.length, k=5)
        self.ndcg10 = ndcg(self.outputs, self.negative, length=self.length, k=10)
        self.ndcg20 = ndcg(self.outputs, self.negative, length=self.length, k=20)

    def _hrat(self):
        self.hrat1 = hr(self.outputs, self.negative, length=self.length, k=1)
        self.hrat5 = hr(self.outputs, self.negative, length=self.length, k=5)
        self.hrat10 = hr(self.outputs, self.negative, length=self.length, k=10)
        self.hrat20 = hr(self.outputs, self.negative, length=self.length, k=20)

    def _mrr(self):
        self.mrr = mrr(self.outputs, self.negative, length=int(self.rating.shape[0]))

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging,
                                            name='first'))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=False,
                                            logging=self.logging))
        self.layers.append(SimpleAttLayer(attention_size=32,
                                          time_major=True))
        self.layers.append(RateLayer(placeholders=self.placeholders,
                                     user_dim=int(self.rating.shape[0]),
                                     item_dim=int(self.rating.shape[1])))


