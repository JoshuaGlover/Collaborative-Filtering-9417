import sys
import math
import time
import numpy   as np
import pandas  as pd
import scipy   as sp
import sgd_svd as ss
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import StratifiedKFold

def rmse(pred, orig):
    # Keep only ratings existent in original dataset and flatten
    pred = pred[orig.nonzero()].flatten()
    orig = orig[orig.nonzero()].flatten()
    # Take mean of difference in ratings and square-root to obtain RMSE
    return math.sqrt(mean_squared_error(pred, orig))

def fetch_dataset(dataset):
    # Load ratings and movie information
    if dataset == "1M":
        print("Using 1M MovieLens dataset")
        ratings = pd.read_csv('movielens_1M/ratings.dat', sep="::", names=['userId', 'movieId', 'rating', 'timestamp'])
        movies  = pd.read_csv('movielens_1M/movies.dat',  sep="::", names=['movieId', 'title', 'genres'])
    else:
        print("Using 100K MovieLens dataset")
        ratings = pd.read_csv('movielens_100k/ratings.csv', usecols=['userId', 'movieId', 'rating', 'timestamp'])
        movies  = pd.read_csv('movielens_100k/movies.csv',  usecols=['movieId', 'title', 'genres'])

    # Count number of users and movies
    num_users  = len(ratings.userId.unique())
    num_movies = len(movies.movieId.unique())
    print(f"Number of Users: {num_users} | Number of Movies {num_movies}")

    # Check sparsity of dataset
    sparsity = round(1.0 - len(ratings) / float (num_users * num_movies), 3)
    print("Sparsity: " + str(sparsity))

    return ratings

def split_dataset(ratings, k):
    # Construct rating dataframe
    rating_frame = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    # Convert to numpy array
    rating_matrix = ratings.values
    # Extract user column
    user_col = ratings['userId'].values

    # Training and test indexes for each fold
    train_matrices = []
    test_matrices = []
    # Split in to training and test sets
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, test_index in skf.split(np.zeros(len(user_col)), user_col):
        # Initialise training and test dataframes
        train_frame = rating_frame.copy()
        test_frame = rating_frame.copy()

        # Set ratings not included to 0 in respective sets
        for user, row in zip(user_col[test_index], rating_matrix[test_index]):
            movie = row[1]
            train_frame[movie][user] = 0

        for user, row in zip(user_col[train_index], rating_matrix[train_index]):
            movie = row[1]
            test_frame[movie][user] = 0

        train_matrices.append(train_frame.values)
        test_matrices.append(test_frame.values)

    return train_matrices, test_matrices


################
# Main Testing #
################

# Check command line arguments
if len(sys.argv) != 4:
    print("Usage: python3 testbench.py <dataset> <num_splits> <num_k_features>")
    sys.exit()

# Dataset name
dataset = sys.argv[1]
# Number of splits for StratifiedKFold
num_splits = int(sys.argv[2])
# Number of latent features used during SVD approximation
k = int(sys.argv[3])

# Fetch dataset and split in to K fold training/test sets
ratings = fetch_dataset(dataset)
train_matrices, test_matrices = split_dataset(ratings, num_splits)

# List of RMSEs from each fold
rmses = []

for train_matrix, test_matrix in zip(train_matrices, test_matrices):
    # Make predictions using SGD SVD
    mf = ss.SVD(train_matrix, k_factors=k, learning_rate=0.002, regularization=0.02, n_epochs=10, min_rating=1, max_rating=5)

    # Train SGD SVD Model
    start_time = time.time()
    mf.train()
    print("Train time: ", (time.time() - start_time))

    # Retrive the fully predicted rating matrix
    pred = mf.full_matrix()

    # Print error and log result of kth fold
    rmse_error = rmse(pred, test_matrix)
    rmses.append(rmse_error)
    print('RMSE: ' + str(rmse_error))

# Print the mean of the RMSEs
print("Mean RMSE: ", (sum(rmses)/len(rmses)))
