import sklearn.datasets
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import multiprocessing as mp
import pickle

## define a class for linear regretion
class LinearRegression():
    """
    n_iters:            iteration times
    learning_rate:      learning rate
    alpha:              alpha for L2 regularization
    tolerant:           minimum tolerant error
    batch_size:         batch size
    """
    def __init__(self, n_iters, learning_rate, alpha, tolerant, batch_size):
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.tolerant = tolerant
        self.batch_size = batch_size
        
    def init_weights(self, n_features):
        limit = np.sqrt(1 / n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, 1))
    
    def prediction(self, X):
        return X.dot(self.w)
    
    ## L2 regularization
    def regularization(self):
        loss = self.w.T.dot(self.w)
        return 0.5 * self.alpha * float(loss)
        
    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.init_weights(n_features)
        self.training_errors = []
        error = np.mean(0.5 * (self.prediction(X) - y) ** 2)
        print(error)
        self.training_errors.append(error)
        
        for i in range(self.n_iters):
            ## shuffle the training data at each iter
            state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(state)
            np.random.shuffle(y)
            for offset in range(0, m_samples, self.batch_size):
                self.fit_batch(offset, X, y)
            error = np.mean(0.5 * (self.prediction(X) - y) ** 2)
            print(error)
            self.training_errors.append(error)
            # print('iter: ',error)
            if error < self.tolerant:
                print('break')
                break

    def fit_batch(self, offset, X, y):
    	end = offset + self.batch_size
    	x_batch = X[offset:end]
    	y_batch = y[offset:end]
    	y_pred = self.prediction(x_batch)
    	# Calculate the loss
    	error = np.sum(0.5 * (y_pred - y_batch) ** 2)
    	# loss = error + self.regularization()
    	# print('batch: ', error, loss)
    	# Calculate the gradient
    	w_grad = x_batch.T.dot(y_pred - y_batch)# + self.alpha * self.w
    	# Update the weight
    	self.w -= self.learning_rate * w_grad

def seperate_random_pandas(percent, data):
    rand_data = data.sample(frac=1.0)
    rand_data = rand_data.reset_index(drop=True)
    wall = int(len(rand_data) * percent)
    data1 = rand_data.loc[0: wall]
    data2 = rand_data.loc[wall+1:]
    return data2, data1

def seperate_random_np(percent, X, Y):
    new_X = copy.deepcopy(X)
    new_Y = copy.deepcopy(Y)
    
    state = np.random.get_state()
    np.random.shuffle(new_X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    
    wall = int(len(new_X) * percent)
    X1 = new_X[0: wall]
    X2 = new_X[wall+1:]
    Y1 = new_Y[0: wall]
    Y2 = new_Y[wall+1:]
    
    return X2, X1, Y2, Y1

## Get all possible combinations of powers for each feature
def sum_combines(powers, max):
	if len(powers) == 1:
		return [[powers[0] + i] for i in range(max-powers[0]+1)]
	else:
		answers = []
		while sum(powers) <= max:
			for item in sum_combines(powers[1:], max-powers[0]):
				item.append(powers[0])
				answers.append(item)
			powers[0] += 1
	return answers

def trans_xi(data, factors, max_grade=4, show=False):
	combines = [1 for i in range(len(factors))]
	power_combinations = sum_combines(combines, max_grade)
	# print(power_combinations)
	ret = np.ones(shape=(len(data), len(power_combinations)), dtype=np.float64)
	for i in range(len(power_combinations)):
		for j in range(len(power_combinations[i])):
			ret[:, i] *= pow(data[factors[j]], power_combinations[i][j])
	return ret

def standardization_pd(data, features, mu, sigma):
	ret_data = copy.deepcopy(data)
	for i in range(len(features)):
		ret_data[features[i]] = (ret_data[features[i]] - mu[i]) / sigma[i] + 1
	return ret_data

## standarization of the xi
def standardization(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float16)

## use Monte-Carlo Cross validation
def train_features(features, house_train, n_iteration):
	mus = []
	sigmas = []
	for feature in features:
		mus.append(house_train[feature].mean())
		sigmas.append(house_train[feature].max() - house_train[feature].min())
	stand_data = standardization_pd(house_train, features, mus, sigmas)
	X = trans_xi(stand_data, features)
	Y = house_train['MedHouseVal'].values.reshape((house_train['MedHouseVal'].values.size, 1))

	error_sum = 0

	for i in range(5):
		x_train, x_valid, y_train, y_valid = seperate_random_np(0.2, X, Y)
		model = LinearRegression(n_iteration, 0.001, 0.5, 0.0001, 128)
		model.fit(x_train, y_train)
		error_sum += np.mean(0.5 * (model.prediction(x_valid) - y_valid) ** 2)

	return error_sum/5

def fit_feature_best(selected_features, feature, data, n_iter):
	now_features = copy.deepcopy(selected_features)
	now_features.append(feature)
	error = train_features(now_features, data, n_iter)
	return error, feature

def find_best_feature(data, n_iter, selected_features, all_features):
	Pool = mp.Pool(8)
	results = []
	for feature in all_features:
		if feature in selected_features:
			continue
		results.append(Pool.apply_async(fit_feature_best, (selected_features, feature, data, n_iter,)))
	Pool.close()
	Pool.join()
	min_error = float('inf')
	for res in results:
		if res.get()[0] < min_error:
			min_error = res.get()[0]
			best_feature = res.get()[1]
	return best_feature, min_error

def fit_feature_least(selected_features, feature, data, n_iter):
	now_features = copy.deepcopy(selected_features)
	now_features.remove(feature)
	error = train_features(now_features, data, n_iter)
	return error, feature

def find_least_feature(data, n_iter, selected_features):
	Pool = mp.Pool(8)
	results = []
	max_error = -float('inf')
	for feature in selected_features:
		results.append(Pool.apply_async(fit_feature_least, (selected_features, feature, data, n_iter, )))

	for res in results:
		if res.get()[0] > max_error:
			max_error = res.get()[0]
			least_feature = res.get()[1]
	return least_feature, max_error

def SFFS(all_features, house_train, max_features=6, iters=1000):
	all_selected_features = [[] for _ in range(max_features+1)]
	selected_features = []
	k = 0
	arg_max = [0 for _ in range(max_features+1)]

	while k < max_features:
		print(k)
		best_feature, min_error = find_best_feature(house_train, iters, selected_features, all_features)
		selected_features.append(best_feature)

		if k < 2:
			k += 1
			arg_max[k] = min_error
			all_selected_features[k] = copy.deepcopy(selected_features)
		else:
			least_feature, max_error = find_least_feature(house_train, iters, selected_features)
			if least_feature == best_feature:
				k += 1
				arg_max[k] = min_error
				all_selected_features[k] = copy.deepcopy(selected_features)
			else:
				handle_features = copy.deepcopy(selected_features)
				handle_features.remove(least_feature)
				if max_error < arg_max[k]:
					if k == 2:
						arg_max[k] = max_error
						all_selected_features[k] = copy.deepcopy(selected_features)
						k += 1
					else:
						stop = False
						while not stop:
							feature_s, error_s = find_least_feature(house_train, iters, features)
							if error_s >= arg_max[k-1]:
								selected_features = copy.deepcopy(handle_features)
								arg_max[k] = max_error
								all_selected_features[k] = copy.deepcopy(selected_features)
								stop = True
							else:
								features.remove(feature_s)
								k -= 1
								if k == 2:
									selected_features = copy.deepcopy(handle_features)
									stop = True
				else:
					k += 1
					arg_max[k] = min_error
					all_selected_features[k] = copy.deepcopy(selected_features)
	return all_selected_features, arg_max

if __name__ == '__main__':
	all_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

	print("Downloading the data...")
	price = sklearn.datasets.fetch_california_housing(as_frame=True)
	print("Finish!")

	house = price['frame']
	print(house.info())
	print(house.describe())

	house_train, house_test = seperate_random_pandas(0.2, house)

	# iters = 300
	# max_features = 4
	# all_selected_features, arg_error = SFFS(all_features, house_train, max_features, iters)
	# print(all_selected_features) 
	# # [['MedInc'], ['MedInc', 'Population'], ['MedInc', 'Population', 'AveBedrms'], ['MedInc', 'Population', 'AveBedrms', 'AveRooms']]
	# print(arg_error)
	# # [0.6609967278780459, 0.6614355403038658, 0.6654750112455751, 0.6968490754079162]


	selected_features = ['MedInc', 'Population']
	mus = []
	sigmas = []
	for feature in selected_features:
		mus.append(house_train[feature].mean())
		sigmas.append(house_train[feature].max() - house_train[feature].min())

	stand_house = standardization_pd(house_train, selected_features, mus, sigmas)
	X = trans_xi(stand_house, selected_features)
	Y = house_train['MedHouseVal'].values.reshape((house_train['MedHouseVal'].values.size, 1))
	n_iteration = 5000

	model = LinearRegression(n_iteration, 0.00001, 0.5, 0.0001, 128)
	model.fit(X, Y)

	stand_house_test = standardization_pd(house_test, selected_features, mus, sigmas)
	x_test = trans_xi(stand_house_test, selected_features)
	y_test = house_test['MedHouseVal'].values.reshape((house_test['MedHouseVal'].values.size, 1))

	y_pred = model.prediction(x_test)

	error = np.mean(0.5*(y_pred - y_test)**2)
	print(error) #0.353289660557462
	data = [model.w, model.training_errors, mus, sigmas]

	with open("./model.pickle", 'wb') as f:
		pickle.dump(data, f)