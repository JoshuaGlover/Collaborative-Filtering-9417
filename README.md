# Collaborative Filtering: Matrix Factorisation vs Neural Networks

All code should be run using Python 3. Dependencies and requirements are listed at the end of this README.

Link to datasets in Google Drive:


Replace the existing "data" folder with the one extracted from the Google Drive.

## Contents

* matrix_factorisation: contains the matrix factorisation implementation
* neural_network: contains the neural network implementation

## Matrix Factorisation Contents

* sgd_svd.py: contains the implementation for the model and training using stochastic gradient descent
* basic_training.py: allows for training the using k fold cross validation for testing RMSE. Provides control over which dataset to use, the number of folds to split the dataset in to, the number of latent features to use, and the number of epochs to train for. Usage: python3 testbench.py <dataset> <num_splits> <num_k_features> <n_epochs>
* rmse_vs_epochs.py: test for rmse as number of training epochs increases
* rmse_vs_k.py: test for rmse as k number of latent features increases
* time_vs_input_size.py: test for the training time as training data size increases. Uses the 1M MovieLens dataset
* example_output: contains text files with examples command line arguments and corresponding outputs for each test file

Note that using the 10M and 20M datasets will require a few minutes for the dataset to be fetched and processed for testing.

## Neural Network Contents

* rs.py: implementation of neural network recommender system
* results.txt

## Requirements

Latest versions of:
* Sklearn
* Numpy
* Pandas
* Numba
* Matplotlib
* Tensorflow
