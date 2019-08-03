# Neural network  for predict recomender
#NOTE: this should probabyl used for a reference only
#Note if you do run it, yo may need to change your pandas view settings
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from keras.layers import Embedding, Reshape, Concatenate,dot,Input,Dense
from keras.models import Model,clone_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
pd.set_option('display.max_columns', 10)

# Load ratings Change directory where appropriate
directory = 'ml-latest-small/'
ratings = pd.read_csv(directory+'ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies = pd.read_csv(directory+'movies.csv', usecols=['movieId', 'title', 'genres'])
max_userid = ratings['userId'].drop_duplicates().max()+1
max_movieid = ratings['movieId'].drop_duplicates().max() +1

#this option is for either training to get weights,to get recomendations,get the whole matrix of predcitons
#or get the RMSE to comapre to et orginal ratings
#training : 0, recomend :1, matrix :2, RMSE :3
mode = 3


#getting values
user_ids = ratings['userId'].values
movie_ids = ratings['movieId'].values
ratings_list = ratings['rating'].values
factors = 100

#making the making teh model

# This is the embedding layer that creates a by latent factors matrix, for the users
user_input = Input(shape=(1,), name = 'user_layers')
user_middle  = Embedding(max_userid, factors,input_length=1)(user_input)
user_layers  = Reshape(target_shape=(factors,))(user_middle)


# This is the embedding layer that creates a latent factors matrix,f or the movies
movie_input = Input(shape=(1,), name = 'movie_layers')
movie_middle  = Embedding(max_movieid, factors,input_length=1)(movie_input)
movie_layers  = Reshape(target_shape=(factors,))(movie_middle)

output = dot([user_layers,movie_layers], axes=1)
model = Model(inputs=[user_input,movie_input], outputs=output)
	
def predict_rating(user_id, movie_id,trained_model):
		return trained_model.predict([np.array([user_id]), np.array([movie_id])])[0][0]
	
#given a user, give a list of movies
def recommedations(user_id,trained_model,ratings=ratings,amount=20):
	user_ratings = ratings[ratings['userId'] == rand_user][['movieId']]
	recommendations = ratings[ratings['movieId'].isin(user_ratings['movieId']) == False][['movieId']].drop_duplicates()
	recommendations['prediction'] = recommendations.apply(lambda x: predict_rating(user_id, x['movieId'],trained_model), axis=1)
	recommendations.sort_values(by='prediction',
		ascending=False,inplace=True)
	recommedations = recommendations.merge(movies,on='movieId',how='inner',
		suffixes=['_u', '_m'])
	recommedations.drop(columns='movieId',inplace=True)
	return recommedations.head(amount)

#training!
if (mode == 0):
	training_model = clone_model(model)

	training_model.compile(loss='mse', optimizer='adamax')

	# Callbacks that monitor the validation loss
	# save the model weights each time the validation loss has improved
	callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint('movie_weights.h5', save_best_only=True)]

	#fitting  the model, i.e. the long part. Will save te best weightsto weights.h5
	history = model.fit([user_ids,movie_ids], ratings_list, epochs=30, validation_split=.1, verbose=2, callbacks=callbacks)
else:
	trained_model = clone_model(model)
	# Load weights if they exists
	
	trained_model.load_weights('movie_weights.h5',by_name=True)

	#given a user, predict the rating for the a movie
	#recomendations!
	if(mode == 1):
		
		rand_user = 450
		print(recommedations(rand_user,trained_model))
		
	elif(mode == 2):
		#the actually computatibley intesive thing to make a matrix equivallent to te one in SVD
		matrix = np.zeros((movie_ids.size,1))
		#print(matrix)
		for user in user_ids:
			newcol = np.array([predict_rating(user, movie,trained_model) for movie in movie_ids])
			newcol.shape=((movie_ids.size,1))
			matrix = np.hstack((matrix,newcol))
		matrix = matrix[:,1:]

		with open('final_matrix.txt','wb') as f:
			for row in matrix:
				np.savetxt(f, row, fmt='%.2f')
	else:
		#index does nothing, but DON'T REMOVE IT
		pred = [predict_rating(row['userId'], row['movieId'],trained_model)for index, row in ratings.iterrows()]
		rmse = math.sqrt(mean_squared_error(pred, ratings_list))
		print( 'RMSE is: ' + '{:.4f}'.format(rmse))

