import sys
import math
import time
import numpy   as np
import sgd_svd as ss
from dataset                 import fetch_dataset
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import StratifiedKFold

if len(sys.argv) != 5:
    print("Usage: python3 testbench.py <dataset> <num_splits> <num_k_features> <n_epochs>")
    sys.exit()

dataset    = sys.argv[1]
n_splits   = int(sys.argv[2])
k_features = int(sys.argv[3])
n_epochs   = int(sys.argv[4])

train_sets, test_sets, user_list, movie_list, B = fetch_dataset(dataset, n_splits)
train_set = train_sets[0]
test_set  = test_sets[0]

# Create matrix factorisation model
mf = ss.SVD(train_set, k_features, user_list, movie_list, B, learning_rate=0.002, \
             regularization=0.02, n_epochs=1, min_rating=1, max_rating=5)

rmses = []
for epoch in range(n_epochs):
    mf.train()

    predictions = []
    originals   = []

    # Calculate RMSE
    for i in range(test_set.shape[0]):
        user, movie, rating = int(test_set[i, 0]), int(test_set[i, 1]), test_set[i, 2]
        pred = mf.user_movie_rating(user, movie)
        predictions.append(pred)
        originals.append(rating)

    rmse = math.sqrt(mean_squared_error(predictions, originals))
    print(rmse)
    rmses.append(rmse)

print(f"\nMinimum RMSE: {np.min(rmses)} at Epoch {np.argmin(rmses)+1}")