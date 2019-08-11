import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from random import *
import warnings

movies_df = pd.read_csv('../data/movielens_1M/movies.dat', sep='::', header=None, engine='python')

ratings_df = pd.read_csv('../data/movielens_1M/ratings.dat', sep='::', header=None, engine='python')

movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

movies_df['List Index'] = movies_df.index

merged = movies_df.merge(ratings_df, on='MovieID')

merged = merged.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

user_rating = merged.groupby('UserID')
print('The Number of Users in Dataset', len(user_rating))
amountOfUsedUsers = 6040

training = []
test = []
loc = 0
for userID, current in user_rating:

    temp = [0]*len(movies_df)

    for num, movie in current.iterrows():
        rand = randint(1, 1000)
        if (rand % 500 == 0) and movie['Rating'] != 0 :
            test_rat = [0] * 3
            test_rat[0] = loc
            test_rat[1] = movie['List Index']
            test_rat[2] = movie['Rating']
            test.append(test_rat)
            movie['Rating'] = 0
        temp[movie['List Index']] = movie['Rating']/5.0

    training.append(temp)
    loc += 1
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1
print('Number of tests ', len(test))
hiddenUnits = 100
visibleUnits = len(movies_df)
vb = tf.placeholder(tf.float32, [visibleUnits])
hb = tf.placeholder(tf.float32, [hiddenUnits])
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])

v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))

_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

alpha = 1.0

w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

err = v0 - v1
err_sum = tf.reduce_mean(err*err)

cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

cur_vb = np.zeros([visibleUnits], np.float32)

cur_hb = np.zeros([hiddenUnits], np.float32)
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
prv_vb = np.zeros([visibleUnits], np.float32)
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(training), batchsize), range(batchsize, len(training), batchsize)):
        batch = training[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: training, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()


MSE = 0
print("calculating mse ", len(test))
prev_user = -1
rec = []
for user in test:
    if prev_user == user[0]:
        rec_score = rec[0][user[1]]
        MSE += ((rec_score) - user[2]/5.0) ** 2
        print(MSE)
        continue
    else:
        inputUser = [training[user[0]]]

        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

        rec_score = rec[0][user[1]]

        MSE += ((rec_score) - user[2]/5.0) ** 2
        # print(MSE)
        prev_user = user[0]
RMSE = math.sqrt(MSE / len(test))
print('RMSE: ', RMSE)
