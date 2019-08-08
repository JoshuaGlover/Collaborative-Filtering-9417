# Neural network  for predict recomender
#NOTE: this should probabyl used for a reference only
#Note if you do run it, yo may need to change your pandas view settings
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from keras.layers import Embedding, Reshape,dot,Input,Dense
from keras.models import Model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
pd.set_option('display.max_columns', 10)

# Load ratings Change directory where appropriate
directory = 'ml-latest/'
ratings = pd.read_csv(directory+'ratings.csv', usecols=['userId', 'movieId', 'rating'])
movies = pd.read_csv(directory+'movies.csv', usecols=['movieId', 'title', 'genres'])
max_userid = ratings['userId'].drop_duplicates().max()+1
max_movieid = ratings['movieId'].drop_duplicates().max() +1

#this option is for either training to get weights,to get recomendations,get the whole matrix of predcitons
#or get the RMSE to comapre to et orginal ratings
#training : 0, recomend :1, matrix :2, RMSE :3
mode = 0


#getting values
user_ids = ratings['userId'].values
movie_ids = ratings['movieId'].values
ratings_list = ratings['rating'].values
factors = 100

#making the making teh model

def model_maker(max_userid = max_userid,max_movieid = max_movieid,factors=factors):
# user_layers is the embedding layer that creates an User by latent factors matrix.

    user_input = Input(shape=(1,), name = 'user_layers')
    user_middle  = Embedding(max_userid, factors,input_length=1)(user_input)
    user_layers  = Reshape(target_shape=(factors,))(user_middle)


    # movie_layers is the embedding layer that creates a Movie by latent factors matrix.
    movie_input = Input(shape=(1,), name = 'movie_layers')
    movie_middle  = Embedding(max_movieid, factors,input_length=1)(movie_input)
    movie_layers  = Reshape(target_shape=(factors,))(movie_middle)

    output = dot([user_layers,movie_layers], axes=1)

    model = Model(inputs=[user_input,movie_input], 
                      outputs=output)
    model.compile(loss='mse', optimizer='adamax')

    return model
	
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
	#split the dataset into a training and test set
	#noramlly random, but its for reprodcbility
	kfold = KFold(n_splits=7, shuffle=True, random_state=np.random.seed(10))
	scores = []
	rmses = []
	min_epochs = []
	i = 0
	for train, test in kfold.split(ratings_list):
		i+=1 
		print("Fold {}".format(i))
		weight_string = 'weights'+str(i)+'.h5'
		model = model_maker()
		callbacks = [EarlyStopping('val_loss', patience=2), ModelCheckpoint(weight_string, save_best_only=True)]
		history = model.fit([user_ids[train],movie_ids[train]],ratings_list[train], epochs=30, validation_split=.1, verbose=2, callbacks=callbacks)
		  
		min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
		rmses.append(math.sqrt(min_val_loss))
		min_epochs.append(idx)
		print( 'Minimum RMSE at epoch ' + '{:d}'.format(idx+1) + ' = '+ '{:.4f}'.format(math.sqrt(min_val_loss)))
		score = model.evaluate([user_ids[test],movie_ids[test]],ratings_list[train])
		print('Loss is: {}'.format(score))
		scores.append(score)
	print("{:.4f}% (+/- {:.4f}%)".format(np.mean(scores), np.std(scores)))
	print(scores)

	print("{:.4f} (+/- {:.4f})".format(np.mean(rmses), np.std(rmses)))
	print(rmses)
	#save rmses and scores to csv
	fp = open("output.csv",'w')
	#print(type(fp))
	fp.write('fold,min_epoch,rmse,loss_score'+'\n')
	for i in range(len(scores)):
		fp.write(str(i)+','+str(min_epochs[i])+','+str(rmses[i])+','+str(scores[i])+','+'\n')
	fp.close()
else:
	trained_model = model_maker()
	# Load weights from best weights by traing
	
	trained_model.load_weights('weights5.h5',by_name=True)

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

