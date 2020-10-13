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
        self.test = None
        self.alphas = None

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
                # self.test = self.layers[i].test
                self.test = hidden
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self._loss()

    def _loss(self):
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
    def __init__(self, placeholders, input_dim, tag, length, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features_'+tag]
        self.input_dim = input_dim
        self.output_dim = FLAGS.output_dim
        self.placeholders = placeholders
        self.tag = tag
        self.length = length

        self.build()

    def _loss(self):
        # Weight decay loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            length=self.length,
                                            placeholders=self.placeholders,
                                            tag=self.tag,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging,
                                            name='first'+self.tag,
                                            featureless=False))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            length=self.length,
                                            placeholders=self.placeholders,
                                            tag=self.tag,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=self.output_dim,
                                            length=self.length,
                                            placeholders=self.placeholders,
                                            tag=self.tag,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))
        self.layers.append(SimpleAttLayer(attention_size=32,
                                          tag=self.tag,
                                          time_major=False))

class AGCNrec():
    def __init__(self,placeholders,input_dim_user,input_dim_item, user_dim, item_dim, learning_rate):
        self.placeholders = placeholders
        self.negative = placeholders['negative']
        self.length = user_dim
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.userModel = GCN(placeholders=self.placeholders, input_dim=input_dim_user, tag='user', length=user_dim)
        self.itemModel = GCN(placeholders=self.placeholders, input_dim=input_dim_item, tag='item', length=item_dim)
        self.user = self.userModel.outputs
        self.item = self.itemModel.outputs
        self.layers = []
        self.rate_matrix = None
        self.result = None
        self.los = 0
        self.hrat1 = 0
        self.hrat5 = 0
        self.hrat10 = 0
        self.hrat20 = 0
        self.ndcg5 = 0
        self.ndcg10 = 0
        self.ndcg20 = 0
        self.mrr = 0
        self.err = None
        self.auc = 0
        # self.mse = 0
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = None

        self.build()

    def build(self):
        self.layers.append(RateLayer(self.user, self.item, user_dim=self.user_dim,item_dim=self.item_dim))
        output = None
        for i in range(len(self.layers)):
            output = self.layers[i]()
        self.rate_matrix = output
        # topk = tf.nn.top_k(test, k).indices
        self.loss()

        self.train()
        self.env()

    def train(self):
        self.train_op = self.optimizer.minimize(self.los)

    def env(self):
        self.result = tf.nn.top_k(self.rate_matrix, 10).indices
        self.hrat()
        self.ndcg()
        self.mr()
        self.au()
        # self.ms()

    def loss(self):
        rating_matrix = self.placeholders['rating']
        self.los += self.userModel.loss
        self.los += self.itemModel.loss
        for i in range(len(self.layers)):
            for var in self.layers[i].vars.values():
                self.los += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.los += tf.losses.mean_squared_error(rating_matrix, self.rate_matrix)
        # self.los += tf.reduce_mean(rating_matrix*tf.log(rating_matrix)-rating_matrix*tf.log(self.rate_matrix))

    def hrat(self):
        self.hrat1 = hr(self.rate_matrix, self.negative, self.length, k=1)
        self.hrat5 = hr(self.rate_matrix, self.negative, self.length, k=5)
        self.hrat10 = hr(self.rate_matrix, self.negative, self.length, k=10)
        self.hrat20 = hr(self.rate_matrix, self.negative, self.length, k=20)

    def ndcg(self):
        self.ndcg5 = ndcg(self.rate_matrix, self.negative, self.length, k=5)
        self.ndcg10 = ndcg(self.rate_matrix, self.negative, self.length, k=10)
        self.ndcg20 = ndcg(self.rate_matrix, self.negative, self.length, k=20)

    def mr(self):
        self.mrr = mrr(self.rate_matrix, self.negative, self.length)

    def au(self):
        self.auc = auc(self.rate_matrix, self.negative, self.length)