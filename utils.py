import numpy as np
import pickle as pkl
import random
import scipy.sparse as sp
import json

# load data and preprocessing
def load_data(user,item):

    support_user = []
    support_item = []
    # rating matrix
    with open('./data/rate_matrix.p','rb') as source:#换成rate_matrix要加todense()，换回来要删去todense()
        rating = pkl.load(source).todense()
    # concept w2v features
    with open('./data/concept_embedding.p','rb') as source:
        concept_w2v = np.array(pkl.load(source))
    # concept bow features
    with open('./data/concept_feature_bow.p','rb') as source:
        concept_bow = pkl.load(source).todense()
    concept = np.hstack((concept_w2v, concept_bow))
    # concept = concept_w2v
    features_item = preprocess_features(concept.astype(np.float32))
    # user features
    with open('./data/UC.p','rb') as source:
        features = pkl.load(source).todense()
    features_user = preprocess_features(features.astype(np.float32))
    # uku
    if 'uku' in user or 'kuk' in item:
        with open('./data/adjacency_matrix.p','rb') as source:
            uk = pkl.load(source).todense()
        if 'uku' in user:
            uk_user = uk.dot(uk.T) + np.eye(uk.shape[0])
            uku = preprocess_adj(uk_user)
            support_user.append(uku)
        if 'kuk' in item:
            ku_item = uk.T.dot(uk) + np.eye(uk.T.shape[0])
            kuk = preprocess_adj(ku_item)
            support_item.append(kuk)
    # ucu
    if 'ucu' in user:
        with open('./data/UC.p', 'rb') as source:
            uc = pkl.load(source).todense()
        uc = uc.dot(uc.T) + np.eye(uc.shape[0])
        ucu = preprocess_adj(uc)
        support_user.append(ucu)
    # uctcu
    if 'uctcu' in user:
        with open('./data/UCT.p','rb') as source:
            uct = pkl.load(source).todense()
        uct = uct.dot(uct.T) + np.eye(uct.shape[0])
        uctcu = preprocess_adj(uct)
        support_user.append(uctcu)
    # uvu
    if 'uvu' in user:
        with open('./data/UV.p','rb') as source:
            uv = pkl.load(source).todense()
        uv = uv.dot(uv.T) + np.eye(uv.shape[0])
        uvu = preprocess_adj(uv)
        support_user.append(uvu)
    # negative sample
    with open('./data/negative.p', 'rb') as source:
        negative = pkl.load(source)
    support_user = np.array(support_user)
    support_item = np.array(support_item)
    return rating, features_item, features_user, support_user, support_item, negative

def preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adjacency):
    rowsum = np.array(adjacency.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adjacency.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)*1e2

def construct_feed_dict(placeholders, features_user, features_item, rating, biases_list_user,
                        biases_list_item, negative):
    feed_dict = dict()
    feed_dict.update({placeholders['rating']: rating})
    feed_dict.update({placeholders['features_user']: features_user})
    feed_dict.update({placeholders['features_item']: features_item})
    feed_dict.update({placeholders['support_user'][i]: biases_list_user[i] for i in range(len(biases_list_user))})
    feed_dict.update({placeholders['support_item'][i]: biases_list_item[i] for i in range(len(biases_list_item))})
    feed_dict.update({placeholders['negative']: negative})
    return feed_dict

def radom_negative_sample(user_action,item_size):
    negative_sample = []
    for u in user_action:
        sample = []
        i = 0
        while i < 99:
            t = random.randint(0,item_size-1)
            if t not in user_action[u]:
                sample.append([u, t])
                i += 1
        sample.append([u, user_action[u][-1]])
        negative_sample.append(sample)
    return np.array(negative_sample)

def getRateMatrix(user_action,item_size):
    row = []
    col = []
    dat = []
    for u in user_action:
        ls = set(user_action[u])
        for k in ls:
            row.append(u)
            col.append(k)
            dat.append(user_action[u].count(k))
    coo_matrix = sp.coo_matrix((dat,(row,col)),shape=(len(user_action),item_size))
    with open('./data/rate_matrix_new.p','wb') as source:
        pkl.dump(coo_matrix.toarray(),source)

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):

            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0