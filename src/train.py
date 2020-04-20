"""
This script trains the SVD model to predict user ratings from the MovieLens datatset. The dataset
size, number of splits user in cross validation, number of latent features used in the model and
the number of training epochs can be set using the following:

Usage: python train.py <dataset> <n_splits> <n_features> <n_epochs>
"""
import svd
import time
import argparse
from math import sqrt
from dataset import fetch_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Process command line arguments
parser = argparse.ArgumentParser(description="Trains SVD Model on MovieLens Dataset")
parser.add_argument("dataset", type=str, help="MovieLens dataset to train on")
parser.add_argument("n_splits", type=int, help="Number of splits for cross validation")
parser.add_argument("n_features", type=int, help="Number of features to be used in model")
parser.add_argument("n_epochs", type=int, help="Number of training epochs to perform")
args = parser.parse_args()

if args.dataset not in ["100K", "1M", "10M", "20M"]:
    parser.error("Dataset does not exist")

dataset = args.dataset
n_splits = args.n_splits
n_features = args.n_features
n_epochs = args.n_epochs

# Fetch the training set, test set, list of users and movies and global bias from the dataset
train_sets, test_sets, user_list, movie_list, global_bias = fetch_dataset(dataset, n_splits)

rmses = []
# Train and measure the error for a new model for each (train_set, test_set) pair (or fold)
for (train_set, test_set) in zip(train_sets, test_sets):
    model = svd.SVD(train_set, n_features, user_list, movie_list, global_bias, \
        learning_rate=0.002, reg_factor=0.02, n_epochs=n_epochs, min_rating=1, max_rating=5)

    # Initialise the timer
    start_time = time.time()

    # Train the model
    model.train()

    # Print training time and fold number to the console
    time_postamble = "Train time: {}s".format(time.time() - start_time)
    print("-"*len(time_postamble))
    print("Fold:", len(rmses)+1)
    print(time_postamble)

    # Obtain all the predicted ratings for the user, movie pairs in the test set
    predicted_ratings, true_ratings = [], []
    for i in range(test_set.shape[0]):
        user, movie, rating = int(test_set[i, 0]), int(test_set[i, 1]), test_set[i, 2]
        predicted_rating = model.compute_rating(user, movie, convert=True)

        # Append predicted and true ratings
        predicted_ratings.append(predicted_rating)
        true_ratings.append(rating)

    # Calculated and print RMSE for the current fold
    rmse = sqrt(mean_squared_error(predicted_ratings, true_ratings))
    print("RMSE:", rmse)
    rmses.append(rmse)

# Calculate and print the final Mean RMSE
print("\nMean RMSE:", sum(rmses)/len(rmses))
