from utils import *
import tensorflow as tf
from models import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('output_dim', 16, 'Output_dim of user final embedding.')
flags.DEFINE_integer('latent_dim', 1024,'Latent_dim of user&item.')

# Load data
rating, concept, features, uk, negative, uc = load_data()

# Some preprocessing
support = [uk]
num_support = len(support)

# Define placeholders
placeholders = {
    'rating': tf.placeholder(dtype=tf.float32,shape=rating.shape,name="rating"),
    'features': tf.placeholder(dtype=tf.float32,shape=features.shape,name='features'),
    'concept': tf.placeholder(dtype=tf.float32,shape=concept.shape,name="concept"),
    'support': [tf.placeholder(dtype=tf.float32,name='support'+str(_)) for _ in range(num_support)],
    'dropout': tf.placeholder_with_default(0.,shape=(),name='dropout'),
    'negative': tf.placeholder(dtype=tf.int32, shape=negative.shape, name='negative')
}

# Create model
model = GCN(placeholders=placeholders, input_dim=features.shape[1], num_support=num_support)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

# Train model
for epoch in range(FLAGS.epochs):

    # Construct feed dictionary
    feed_dict = construct_feed_dict(placeholders,features,rating,concept,support, negative)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    _, loss, hrat1, hrat5, hrat10, hrat20, ndcg5, ndcg10, ndcg20, mrr, auc = sess.run([model.opt_op, model.loss,
                                                                                       model.hrat1, model.hrat5, model.hrat10,
                                                                                       model.hrat20, model.ndcg5, model.ndcg10,
                                                                                       model.ndcg20, model.mrr, model.accuracy],
                                                                                      feed_dict=feed_dict)
    if epoch%50 == 0:
        print('Train:'+str(epoch)+' Loss:'+str(loss)+' HR@1:'+str(hrat1)+
              ' HR@5:'+str(hrat5)+' HR@10:'+str(hrat10)+' HR@20:'+str(hrat20)+
              ' nDCG5:'+str(ndcg5)+' nDCG10:'+str(ndcg10)+' nDCG20:'+str(ndcg20)+
              ' MRR:'+str(mrr)+' AUC:'+str(auc))