import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from random import *
import warnings

# Load the movies dataset and also pass header=None since files don't contain any headers
movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python')

# Load the ratings dataset
ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python')

# Lets rename our columns in these data frames so we can convey their data better
movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# Data Correction and Formatting


"""
- Our Movie ID's vary from 1 to 3952 while we have 3883 movies. 
- Due to this, we won't be able to index movies through their ID since we would get memory indexing errors. 
- To amend we can create a column that shows the spot in our list that particular movie is in:
"""

movies_df['List Index'] = movies_df.index


# Merge movies_df with ratings_df by MovieID
merged_df = movies_df.merge(ratings_df, on='MovieID')

# Drop unnecessary columns
merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

# Lets Group up the Users by their user ID's
user_Group = merged_df.groupby('UserID')
print('The Number of Users in Dataset', len(user_Group))
"""
Formatting the data into input for the RBM. 
Store the normalized users ratings into a list of lists called trX.
"""

# Amount of users used for training
amountOfUsedUsers = 6040

# Creating the training list
trX = []
test = []
loc = 0
# For each user in the group
for userID, curUser in user_Group:

    # Create a temp that stores every movie's rating
    temp = [0]*len(movies_df)

    # For each movie in curUser's movie list
    for num, movie in curUser.iterrows():
        rand = randint(1, 1000)
        if (rand % 500 == 0) and movie['Rating'] != 0 :
            test_rat = [0] * 3
            test_rat[0] = loc
            test_rat[1] = movie['List Index']
            test_rat[2] = movie['Rating']
            test.append(test_rat)
            movie['Rating'] = 0
            #print("Inserted to test group", test_rat)
        # Divide the rating by 5 and store it, Ratings are normalized between 0 and 1
        temp[movie['List Index']] = movie['Rating']/5.0

    # Add the list of ratings into the training list
    trX.append(temp)
    loc += 1
    # Check to see if we finished adding in the amount of users for training
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1
print('Number of tests ', len(test))
# Setting the models Parameters
hiddenUnits = 50
visibleUnits = len(movies_df)
vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique movies
hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features were going to learn
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix

# Phase 1: Input Processing
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling

# Phase 2: Reconstruction
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

""" Set RBM Training Parameters """

# Learning rate
alpha = 1.0

# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err*err)

""" Initialize our Variables with Zeroes using Numpy Library """

# Current weight
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Current visible unit biases
cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100, After training print out the error by epoch
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()

"""
Recommendation System :-
- We can now predict movies that an arbitrarily selected user might like. 
- This can be accomplished by feeding in the user's watched movie preferences into the RBM and then reconstructing the 
  input. 
- The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched 
  based on the preferences of the users that the RBM was trained on.
"""

# Select the input User

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
        inputUser = [trX[user[0]]]

        # Feeding in the User and Reconstructing the input
        hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
        vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
        feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
        rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

        # scored_movies_df = movies_df
        rec_score = rec[0][user[1]]
        #print('for user ', user[0]) 
        #print ("recommendation score is ", rec_score)
        #print ("actual score is ", user[2])
        MSE += ((rec_score) - user[2]/5.0) ** 2
        print(MSE)
        prev_user = user[0]
RMSE = math.sqrt(MSE / len(test))
print(RMSE)