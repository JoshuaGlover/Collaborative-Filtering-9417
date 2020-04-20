import sys
import math
import time
import numpy   as np
import svd
from dataset                 import fetch_dataset
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# Process command line arguments
if len(sys.argv) != 4:
    print("Usage: python3 rmse_vs_epochs.py <dataset> <n_features> <n_epochs>")
    sys.exit()

dataset    = sys.argv[1]
n_features = int(sys.argv[2])
n_epochs   = int(sys.argv[3])

# Fetch dataset
train_sets, test_sets, user_list, movie_list, B = fetch_dataset(dataset, 5)

# Just use one fold
train_set = train_sets[0]
test_set  = test_sets[0]

# Create matrix factorisation model
model = svd.SVD(train_set, n_features, user_list, movie_list, B, learning_rate=0.002, \
             reg_factor=0.02, n_epochs=1, min_rating=1, max_rating=5)

# Keep track of rmse after every epoch
rmses = []

# Loop number of epochs, measuring RMSE after each
print("RMSEs after every Epoch:\n")
for epoch in range(n_epochs):
    model.train()

    predictions = []
    originals   = []

    # Calculate RMSE
    for i in range(test_set.shape[0]):
        user, movie, rating = int(test_set[i, 0]), int(test_set[i, 1]), test_set[i, 2]
        pred = model.compute_rating(user, movie, convert=True)
        predictions.append(pred)
        originals.append(rating)

    rmse = math.sqrt(mean_squared_error(predictions, originals))
    print(rmse)
    rmses.append(rmse)

# Return the lowest RMSE achieved and at what Epoch it occured
print(f"\nMinimum RMSE: {np.min(rmses)} at Epoch {np.argmin(rmses)+1}")
