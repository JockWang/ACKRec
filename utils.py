import numpy as np
import pickle as pkl
import random

# load data and preprocessing
def load_data():
    with open('./data/rate_matrix.p','r') as source:
        rating = pkl.load(source).todense()
    with open('./data/concept_embedding.p','r') as source:
        concept = np.array(pkl.load(source))
    with open('./data/user_feature.p','r') as source:
        features = pkl.load(source).todense()
    features = preprocess_features(features)
    with open('./data/adjacency_matrix.p','r') as source:
        adjacency = pkl.load(source).todense()
    adjacency = preprocess_adj(adjacency)
    with open('./data/UC.p','r') as source:
        uc = pkl.load(source).todense()
    uc = preprocess_adj(uc)
    with open('./data/user_action.p','r') as source:
        user_action = pkl.load(source)
    negative = radom_negative_sample(user_action,rating.shape[1])
    with open('./data/negative.p', 'w') as source:
        pkl.dump(negative,source)

    return rating,concept,features,adjacency,negative,uc

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adjacency):
    adjacency = adjacency.dot(adjacency.T)
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt

def construct_feed_dict(placeholders, features, rating, concept, support, negative):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['rating']: rating})
    feed_dict.update({placeholders['concept']: concept})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['negative']: negative})
    return feed_dict

def radom_negative_sample(user_action,item_size):
    negative_sample = []
    for u in user_action:
        sample = []
        i = 0
        while i < 99:
            t = random.randint(0,item_size-1)
            if t != user_action[u][-1]:
                sample.append([u, t])
                i += 1
        sample.append([u, user_action[u][-1]])
        negative_sample.append(sample)
    return np.array(negative_sample)

