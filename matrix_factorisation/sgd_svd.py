import math
import time
import numpy as np
from numba import njit
from sklearn.metrics import mean_squared_error

class SVD():
    def __init__(self, samples, k_factors, user_list, movie_list, B, learning_rate=0.002, regularization=0.02, n_epochs=1,
                 min_rating=1, max_rating=5):
        """
        Trains on a input matrix R with the aim of filling in empty values
        using k latent features

        The following sources were used as reference:
        http://www.albertauyeung.com/post/python-matrix-factorization/
        https://sifter.org/~simon/journal/20061211.html
        https://github.com/gbolmier/funk-svd

        Attributes:

        samples    : training set of rating samples
        num_users  : number of unique users
        num_movies : number of unique movies
        k_factors  : number of latent features used
        learning_rate  : learning rate used for SGD
        regularization : regularization parameter used for SGD
        n_epochs   : number of training epochs
        min_rating : minimum rating that can be given (1)
        max_rating : maximum rating that can be given (5)
        P          : matrix of user latent features
        Q          : matrix of movie latent features
        B_u        : rating bias of users
        B_m        : rating bias for movies
        B          : global rating bias/mean
        user_dict  : translates actual userId to index in P
        movie_dict : translates actual movieId to index in Q

        """
        self.samples    = samples
        self.num_users  = len(user_list)
        self.num_movies = len(movie_list)
        self.k_factors  = k_factors
        self.learning_rate  = learning_rate
        self.regularization = regularization
        self.n_epochs   = n_epochs
        self.min_rating = min_rating
        self.max_rating = max_rating

        # Initialise P and Q matrices with random values
        self.P = np.random.normal(0, 1./self.k_factors, (self.num_users, self.k_factors))
        self.Q = np.random.normal(0, 1./self.k_factors, (self.num_movies, self.k_factors))

        # Initialise user and movie biases to zero
        self.B_u = np.zeros(self.num_users)
        self.B_m = np.zeros(self.num_movies)

        # Initialise global bias
        self.B = B

        # User and movie dictionary (translates P and Q indexes to actual IDs)
        self.user_dict  = dict(zip(user_list,  list(range(self.num_users))))
        self.movie_dict = dict(zip(movie_list, list(range(self.num_movies))))

        # Convert training samples
        for i in range(self.samples.shape[0]):
            u, m, rating = int(self.samples[i, 0]), int(self.samples[i, 1]), self.samples[i, 2]
            self.samples[i, 0] = self.user_dict[u]
            self.samples[i, 1] = self.movie_dict[m]

    def train(self):
        for epoch in range(self.n_epochs):
            # print("Epoch: ", epoch)
            self.P, self.Q, self.B_u, self.B_m = fast_sgd(self.samples, self.B, self.B_u, self.B_m, \
                                                          self.P, self.Q, self.learning_rate, self.regularization)
            if (epoch+1) % 10 == 0 or epoch == 0:
                start_time = time.time()
                if epoch == 0:
                    padded_epoch = self.pad(epoch)
                else:
                    padded_epoch = self.pad(epoch+1)
                # print("Epoch: {} | RMSE: {}".format(padded_epoch, self.error()))
                # print("Time:", time.time() - start_time)

    def error(self):
        # Iterate over samples
        predictions = []
        original = []
        for i in range(self.samples.shape[0]):
            user, movie, rating = int(self.samples[i, 0]), int(self.samples[i, 1]), self.samples[i, 2]
            # print("User", user, "Movie", movie)
            pred = self.compute_rating(user, movie)
            predictions.append(pred)
            original.append(rating)

        return math.sqrt(mean_squared_error(predictions, original))

    def compute_rating(self, user, movie):
        # print("U", u, "M", m, "User", user, "Movie", movie)
        pred = self.B + self.B_u[user] + self.B_m[movie] + self.P[user, :].dot(self.Q[movie, :].T)
        return pred

    def user_movie_rating(self, u, m):
        # print("U", u, "M", m, "User", user, "Movie", movie)
        user  = self.user_dict[u]
        movie = self.movie_dict[m]
        pred = self.B + self.B_u[user] + self.B_m[movie] + self.P[user, :].dot(self.Q[movie, :].T)
        np.clip(pred, self.min_rating, self.max_rating)
        return pred

    def pad(self, epoch):
        reps = len(str(self.n_epochs)) - len(str(epoch))
        padded_epoch = "0"*reps + str(epoch)
        return padded_epoch

@njit
def fast_sgd(samples, B, B_u, B_m, P, Q, learning_rate, regularization):
    # Iterate over samples
    for i in range(samples.shape[0]):
        user, movie, rating = int(samples[i, 0]), int(samples[i, 1]), samples[i, 2]

        pred = B + B_u[user] + B_m[movie] + P[user, :].dot(Q[movie, :].T)
        err = rating - pred

        # Adjust biases
        B_u[user] += learning_rate * (err - regularization * B_u[user])
        B_m[movie] += learning_rate * (err - regularization * B_m[movie])

        # Adjust P and Q
        P[user,:] += learning_rate * (err * Q[movie,:] - regularization * P[user,:])
        Q[movie,:] += learning_rate * (err * P[user,:] - regularization * Q[movie,:])

    return P, Q, B_u, B_m
