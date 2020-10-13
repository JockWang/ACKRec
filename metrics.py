import tensorflow as tf

def loss(rating,rate):
    err = tf.square(tf.subtract(rating,rate))
    los = tf.reduce_sum(err)
    return los

def auc(rate, negative, length):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, 100).indices
    where = tf.where(tf.equal(topk, tf.constant(99,shape=[length,100])))
    auc = tf.split(where,num_or_size_splits=2,axis=1)[1]
    ran_auc = tf.Variable(tf.random_uniform(shape=[length, 1], minval=0, maxval=100, dtype=tf.int64))
    auc = tf.reduce_mean(tf.cast(tf.less(auc - ran_auc, 0), dtype=tf.float32))
    return auc

def hr(rate, negative, length, k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices
    isIn = tf.cast(tf.equal(topk, 99), dtype=tf.float32)
    row = tf.reduce_sum(isIn, axis=1)
    all = tf.reduce_sum(row)
    return all/length

def mrr(rate, negative, length):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, 100).indices
    mrr_ = tf.reduce_sum(1 / tf.add(tf.split(value=tf.where(tf.equal(topk, tf.constant(99, shape=[length, 100]))),
                                             num_or_size_splits=2, axis=1)[1], 1))
    mrr = mrr_/length
    return mrr

def ndcg(rate, negative, length, k=5):
    test = tf.gather_nd(rate, negative)
    topk = tf.nn.top_k(test, k).indices
    n = tf.split(value=tf.where(tf.equal(topk, tf.constant(99, shape=[length, k]))), num_or_size_splits=2, axis=1)[1]
    ndcg = tf.reduce_sum(tf.log(2.0) / tf.log(tf.cast(tf.add(n, tf.constant(2, dtype=tf.int64)),
                                                      dtype=tf.float32)))/length
    return ndcg

def env(rate, negative, length):
    hrat1 = hr(rate,negative,length,k=1)
    hrat5 = hr(rate,negative,length,k=5)
    hrat10 = hr(rate,negative,length,k=10)
    hrat20 = hr(rate,negative,length,k=20)
    ndcg5 = ndcg(rate,negative,length,k=5)
    ndcg10 = ndcg(rate,negative,length,k=10)
    ndcg20 = ndcg(rate,negative,length,k=20)
    mr = mrr(rate,negative,length)
    au = auc(rate,negative,length)
    return hrat1, hrat5, hrat10, hrat20, ndcg5, ndcg10, ndcg20, mr, au
