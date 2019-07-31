import svd
import math
import time
import sys
import numpy  as np
import pandas as pd
import scipy  as sp
from sklearn.metrics import mean_squared_error

def rmse(pred, orig):
    # Keep only ratings existent in original dataset and flatten
    pred = pred[orig.nonzero()].flatten()
    orig = orig[orig.nonzero()].flatten()
    # Take mean of difference in ratings and square-root to obtain RMSE
    return math.sqrt(mean_squared_error(pred, orig))

def load_dataset(size):
    # Load ratings and movies from files
    if size == "100k":
        ratings = pd.read_csv('movielens_100k/ratings.csv', usecols=['userId', 'movieId', 'rating', 'timestamp'])
        movies  = pd.read_csv('movielens_100k/movies.csv',  usecols=['movieId', 'title', 'genres'])
    elif size == "1M":
        ratings = pd.read_csv('movielens_1M/ratings.dat', sep="::", names=['userId', 'movieId', 'rating', 'timestamp'])
        movies  = pd.read_csv('movielens_1M/movies.dat',  sep="::", names=['movieId', 'title', 'genres'])
    else:
        print("Dataset doesn't exist")

    # Count number of users and movies
    num_users  = len(ratings.userId.unique())
    num_movies = len(movies.movieId.unique())
    print(f"Number of Users: {num_users} | Number of Movies {num_movies}")

    # Check sparsity of dataset
    sparsity = round(1.0 - len(ratings) / float (num_users * num_movies), 3)
    print("Sparsity: " + str(sparsity))

    # Construct rating dataframe
    rating_frame = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    # Convert to numpy array
    rating_matrix = rating_frame.values
    # Adjust the user ratings by subtracting the mean of their ratings
    user_rating_mean = np.mean(rating_matrix, axis=1)
    ratings_demeaned = rating_matrix - user_rating_mean.reshape(-1, 1)

    # Returned demeaned matrix and original rating matrix
    return ratings_demeaned, rating_matrix, user_rating_mean

########
# Main #
########

if len(sys.argv) != 3:
    print("Usage: python3 testbench.py <dataset> <k>")
    sys.exit()

# K is the number of features we will use
K = int(sys.argv[2])
# Load the dataset
demeaned, orig, user_rating_mean = load_dataset(sys.argv[1])

##############
# Custom SVD #
##############

# Run, time and test our custom svd implementation
start_time = time.time()
U, S, VT = svd.custom_svd(demeaned, K)
print("Custom SVD execution time: " + str(time.time() - start_time) + "s")

# Make predictions and calculate RMSE
pred = np.dot(np.dot(U, S), VT) + user_rating_mean.reshape(-1, 1)
print('Custom RMSE: ' + str(rmse(pred, orig)))

#############
# Scipy SVD #
#############

# Run, time and test our custom svd implementation
start_time = time.time()
U, S, VT = svd.scipy_svd(demeaned, K)
print("Scipy SVD execution time: " + str(time.time() - start_time) + "s")

# Make predictions and calculate RMSE
pred = np.dot(np.dot(U, S), VT) + user_rating_mean.reshape(-1, 1)
print('Scipy RMSE: ' + str(rmse(pred, orig)))
