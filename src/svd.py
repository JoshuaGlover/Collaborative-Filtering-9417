import numpy as np
from math import sqrt
from numba import njit
from sklearn.metrics import mean_squared_error

class SVD():
    """ Singular value decomposition (SVD) using stochastic gradient descent (SGD).

    Trains on a input matrix R with the aim of filling in empty values using latent features.
    The following sources were used as reference:

    http://www.albertauyeung.com/2017/04/23/python-matrix-factorization/
    https://sifter.org/~simon/journal/20061211.html

    Attributes:
        samples: List of trainng samples
        n_samples (int): Number of training samples
        n_users (int): Number of unique users across all samples
        n_items (int): Number of unique items across all samples

        P (ndarray): Matrix containing users and latent features
        Q (ndarray): Matrix containing items and latent features
        B_u (ndarray): Biases for each user
        B_i (ndarray): Biases for each item
        B (int): The global bias or average rating over all ratings
        n_features (int): The number of latent features used for matrices P and Q

        user_dict (dict): Dictionary mapping user IDs in samples to index in P
        item_dict (dict): Dictionary mapping item IDs in samples to index in Q

        learning_rate (float): Controls the speed at which the model learns
        reg_factor (float): A constant that prevents the values of P and Q from becoming large
        n_epochs (int): Number of epochs the model is trained for

        min_rating (int): Minimum rating that can be given
        max_rating (int): Maximum rating that can be given
    """

    def __init__(self, samples, n_features, user_list, item_list, global_bias, learning_rate=0.002,
                 reg_factor=0.02, n_epochs=1, min_rating=1, max_rating=5):
        """ Initialises the model. See help(SVD) for more information. """
        self.samples = samples
        self.n_samples = self.samples.shape[0]
        self.n_users = len(user_list)
        self.n_items = len(item_list)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor
        self.n_epochs = n_epochs
        self.min_rating = min_rating
        self.max_rating = max_rating

        # Initialise P and Q matrices with random values
        self.P = np.random.normal(0, 1./self.n_features, (self.n_users, self.n_features))
        self.Q = np.random.normal(0, 1./self.n_features, (self.n_items, self.n_features))

        # Initialise user and item biases to zero, and initialise global bias
        self.B_u = np.zeros(self.n_users)
        self.B_i = np.zeros(self.n_items)
        self.B = global_bias

        # User and item dictionaries (translates P and Q indexes to actual IDs)
        self.user_dict = dict(zip(user_list, list(range(self.n_users))))
        self.item_dict = dict(zip(item_list, list(range(self.n_items))))

        # Convert training samples to use P and Q indexes rather than original user ID
        for i in range(self.n_samples):
            user, item, rating = int(self.samples[i, 0]), int(self.samples[i, 1]), self.samples[i, 2]
            self.samples[i, 0] = self.user_dict[user]
            self.samples[i, 1] = self.item_dict[item]

    def train(self):
        """ Trains the model using the sample ratings over a set number of epochs. """
        for epoch in range(self.n_epochs):
            # Perform one update using stochastic gradient descent
            self.P, self.Q, self.B_u, self.B_m = sgd(self.samples, self.P, self.Q, self.B, \
                self.B_u, self.B_i, self.learning_rate, self.reg_factor)

    def error(self):
        """
        Calculates and returns the root mean squared error (RMSE) between the predicted and true
        rating over all samples.

        Returns:
            float: The root mean squared error (RMSE)
        """
        predicted_ratings, true_ratings = [], []
        for i in range(self.n_samples):
            user, item, rating = int(self.samples[i, 0]), int(self.samples[i, 1]), self.samples[i, 2] 

            # Append the predicted and true rating
            predicted_ratings.append(self.compute_rating(user, item))
            true_ratings.append(rating)

        # Return the RMSE
        return sqrt(mean_squared_error(predicted_ratings, true_ratings))

    def compute_rating(self, user, item, convert=False):
        """ Computes the predicted rating using the user and item indexes for P and Q.

        Args:
            user (int): The index/row of P corresponding to the user
            item (int): The index/row of Q corresponding to the item
            convert (bool): If true, converts the user and item ID to the matrix indexes for P and Q

        Returns:
            float: The predicted rating using the model

        """
        if convert:
            # Convert IDs from dataset IDs to matrix row indexes
            user = self.user_dict[user]
            item = self.item_dict[item]
        return self.B + self.B_u[user] + self.B_i[item] + self.P[user, :].dot(self.Q[item, :].T)

@njit
def sgd(samples, P, Q, B, B_u, B_i, learning_rate, reg_factor):
    """ Performs one epoch using stochastic gradient descent to update P, Q, B_u and B_i

    Args:
        samples: The training samples containing user, item, rating tuples

        P (ndarray): Matrix containing users and latent features
        Q (ndarray): Matrix containing items and latent features

        B (int): The global bias
        B_u (ndarray): Biases for each user
        B_i (ndarray): Biases for each item

        learning_rate (float): Controls the speed at which the model learns
        reg_factor (float): A constant that prevents the values of P and Q from becoming large

    Returns:
        type: Description of returned object.

    """
    for i in range(samples.shape[0]):
        user, item, rating = int(samples[i, 0]), int(samples[i, 1]), samples[i, 2]

        # Calculate the models predicted value for the user, item pair
        prediction = B + B_u[user] + B_i[item] + P[user, :].dot(Q[item, :].T)
        # Calculate the error as the different between the predicted and true rating
        error = rating - prediction

        # Update P and Q
        P[user,:] += learning_rate * (error * Q[item,:] - reg_factor * P[user,:])
        Q[item,:] += learning_rate * (error * P[user,:] - reg_factor * Q[item,:])

        # Update the user and item biases
        B_u[user] += learning_rate * (error - reg_factor * B_u[user])
        B_i[item] += learning_rate * (error - reg_factor * B_i[item])

    return P, Q, B_u, B_i
