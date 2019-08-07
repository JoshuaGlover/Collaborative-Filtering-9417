import numpy as np

class SVD():
    def __init__(self, R, k_factors, learning_rate=0.002, regularization=0.02, n_epochs=100,
                 min_rating=1, max_rating=5):
        """
        Trains on a input matrix R with the aim of filling in empty values
        using k latent features

        """

        self.R = R
        self.num_users, self.num_movies = R.shape
        self.k_factors = k_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.min_rating = min_rating
        self.max_rating = max_rating

        # Initialise P and Q matrices with random values
        self.P = np.random.normal(0, 1./self.k_factors, (self.num_users, self.k_factors))
        self.Q = np.random.normal(0, 1./self.k_factors, (self.num_movies, self.k_factors))

        # Initialise user and movie biases to zero
        self.B_u = np.zeros(self.num_users)
        self.B_m = np.zeros(self.num_movies)

        # Initialise global bias
        self.B = np.mean(R[R.nonzero()])

        # Create a list of tuples (user, movie, rating) for all nonzero element in R
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_movies)
            if self.R[i, j] != 0
        ]

    def train(self):
        for epoch in range(self.n_epochs):
            print("Epoch: ", epoch)
            # TODO: Potentially implement shuffling
            self.sgd()

    def sgd(self):
        # Iterate over samples
        for i, j, r in self.samples:
            pred = self.compute_rating(i, j)
            err = r - pred

            # Adjust biases
            self.B_u[i] += self.learning_rate * (err - self.regularization * self.B_u[i])
            self.B_m[j] += self.learning_rate * (err - self.regularization * self.B_m[j])

            # Adjust P and Q
            self.P[i,:] += self.learning_rate * (err * self.Q[j,:] - self.regularization * self.P[i,:])
            self.Q[j,:] += self.learning_rate * (err * self.P[i,:] - self.regularization * self.Q[j,:])

    def compute_rating(self, user, movie):
        pred = self.B + self.B_u[user] + self.B_m[movie] + self.P[user, :].dot(self.Q[movie, :].T)
        return pred

    def full_matrix(self):
        matrix = self.B + self.B_u[:,np.newaxis] + self.B_m[np.newaxis:,] + self.P.dot(self.Q.T)
        # Clip between min and max rating
        matrix = np.clip(matrix, self.min_rating, self.max_rating)
        return matrix
