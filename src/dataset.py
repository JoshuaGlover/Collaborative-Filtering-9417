import sys
import math
import time
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

def fetch_dataset(dataset, n_splits, shuffle=True):
    # Fetch dataset
    if dataset == "20M":
        print("Using 20M Dataset")
        names = ['userId', 'movieId', 'rating']
        dtype = {'userId': np.uint32, 'movieId': np.uint32, 'rating': np.float64}
        ratings = pd.read_csv('../data/movielens_20M/ratings.csv', sep=",", names=names, dtype=dtype, \
                              header=None, engine='python', usecols=[0,1,2], skiprows=1)
    elif dataset == "1M":
        print("Using 1M Dataset")
        names = ['userId', 'movieId', 'rating']
        dtype = {'userId': np.uint32, 'movieId': np.uint32, 'rating': np.float64}
        ratings = pd.read_csv('../data/movielens_1M/ratings.dat', sep="::", names=names, dtype=dtype, \
                              header=None, engine='python', usecols=[0,1,2])
    elif dataset == "10M":
        print("Using 10M Dataset")
        names = ['userId', 'movieId', 'rating']
        dtype = {'userId': np.uint32, 'movieId': np.uint32, 'rating': np.float64}
        ratings = pd.read_csv('../data/movielens_10M/ratings.dat', sep="::", names=names, dtype=dtype, \
                              header=None, engine='python', usecols=[0,1,2])
    else:
        print("Using 100K Dataset")
        names = ['userId', 'movieId', 'rating']
        dtype = {'userId': np.uint32, 'movieId': np.uint32, 'rating': np.float64}
        ratings = pd.read_csv('../data/movielens_100K/ratings.csv', sep=",", names=names, dtype=dtype, \
                              header=None, engine='python', usecols=[0,1,2], skiprows=1)

    # Count number of unique users and movies
    num_users  = len(ratings.userId.unique())
    num_movies = len(ratings.movieId.unique())
    print(f"Number of Users: {num_users} | Number of Movies: {num_movies}")
    # Check sparsity of dataset
    sparsity = round(1.0 - len(ratings) / float (num_users * num_movies), 3)
    print("Sparsity: " + str(sparsity))

    # Form a list of unique users and movies
    user_list  = ratings['userId'].unique().tolist()
    movie_list = ratings['movieId'].unique().tolist()

    # Calculate the mean rating from all samples
    B = ratings['rating'].mean()

    # Split the dataset in to training and test sets
    if dataset == "1M" or dataset == "100K":
        # Extract the column of users so that we can split the dataset per user
        user_col = ratings['userId'].values

        # Initialise sets of training and test indexes for each fold
        train_sets = []
        test_sets = []
        # Apply StratifiedKFold to split dataset
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        for train_index, test_index in skf.split(np.zeros(len(user_col)), user_col):
            train_set = (ratings.iloc[train_index]).values
            test_set  = (ratings.iloc[test_index]).values
            train_sets.append(train_set)
            test_sets.append(test_set)

    else:
        # StratifiedKFold takes too long on 10M and 20M dataset so use this instead
        train = ratings.sample(frac=0.8, random_state=7)
        test  = ratings.drop(train.index.tolist())
        train_sets = [train.values]
        test_sets  = [test.values]

    print("Train Set Length:", len(train_sets[0]), "| Test Set Length:", len(test_sets[0]))

    return train_sets, test_sets, user_list, movie_list, B
