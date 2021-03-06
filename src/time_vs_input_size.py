import sys
import math
import time
import numpy   as np
import svd
import pandas  as pd
from dataset                 import fetch_dataset
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# Process command line arguments
if len(sys.argv) != 2:
    print("Usage: python3 time_vs_input_size.py <n_features>")
    sys.exit()

n_features = int(sys.argv[1])

names = ['userId', 'movieId', 'rating']
dtype = {'userId': np.uint32, 'movieId': np.uint32, 'rating': np.float64}
ratings = pd.read_csv('../data/movielens_1M/ratings.dat', sep="::", names=names, dtype=dtype, \
                      header=None, engine='python', usecols=[0,1,2])

# Form a list of unique users and movies
user_list  = ratings['userId'].unique().tolist()
movie_list = ratings['movieId'].unique().tolist()

# Calculate the mean rating from all samples
B = ratings['rating'].mean()

for size in range(1,51):
    curr_test_set = (ratings.sample(frac=(size/50), random_state=7)).values

    # Create matrix factorisation model
    mf = svd.SVD(curr_test_set, n_features, user_list, movie_list, B, learning_rate=0.002, \
                reg_factor=0.02, n_epochs=5, min_rating=1, max_rating=5)

    # Time training and print result
    start_time = time.time()
    mf.train()
    time_postamble = "{:.2f} % of 1M dataset took {:.6f}s".format((size/50)*100, time.time() - start_time)
    print(time_postamble)
