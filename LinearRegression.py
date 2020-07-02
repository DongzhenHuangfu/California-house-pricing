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
        
        for i in range(self.n_iters):
            ## shuffle the training data at each iter
            state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(state)
            np.random.shuffle(y)
            for offset in range(0, m_samples, self.batch_size):
                self.fit_batch(offset, X, y)
            error = np.mean(0.5 * (self.prediction(X) - y) ** 2)
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
    	loss = error + self.regularization()
    	# print('batch: ', error, loss)
    	# Calculate the gradient
    	w_grad = x_batch.T.dot(y_pred - y_batch) + self.alpha * self.w
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

def trans_xi(data, factors, max_grade=4, show=False):
    ret = np.ones(shape=(len(data), pow(max_grade+1, len(factors))), dtype=np.float64)
    if show:
        with tqdm(total=pow(max_grade+1, len(factors))-max_grade-1) as pbar:
            for i in range(len(factors)):
                for grade in range(max_grade+1):
                    if i == 0:
                        if grade != 0:
                            ret[:, grade] = data[factors[i]] * ret[:, grade-1]
                    else:
                        now_vec = ret[:, 0:pow(max_grade+1, i)]
                        if grade != 0:
                            for j in range(pow(max_grade+1, i)):
                                ret[:, grade * pow(max_grade+1, i) + j] = now_vec[:, j] * pow(data[factors[i]], grade)
                                pbar.update(1)
    else:
        for i in range(len(factors)):
            for grade in range(max_grade+1):
                if i == 0:
                    if grade != 0:
                        ret[:, grade] = data[factors[i]] * ret[:, grade-1]
                else:
                    now_vec = ret[:, 0:pow(max_grade+1, i)]
                    if grade != 0:
                        for j in range(pow(max_grade+1, i)):
                            ret[:, grade * pow(max_grade+1, i) + j] = now_vec[:, j] * pow(data[factors[i]], grade)
    return ret

## standarization of the xi
def standardization(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float16)

## use Monte-Carlo Cross validation
def train_features(features, house_train, n_iteration):
	X = trans_xi(house_train, features)
	Y = house_train['MedHouseVal'].values.reshape((house_train['MedHouseVal'].values.size, 1))

	mu = X.mean(axis=0)
	divid = X.max(axis=0) - X.min(axis=0)

	mu[0] = 0
	divid[0] = 1

	X_Stand = standardization(X, mu, divid)
	error_sum = 0

	for i in range(5):
		x_train, x_valid, y_train, y_valid = seperate_random_np(0.2, X_Stand, Y)
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
		else:
			least_feature, max_error = find_least_feature(house_train, iters, selected_features)
			if least_feature == best_feature:
				k += 1
				arg_max[k] = min_error
			else:
				handle_features = copy.deepcopy(selected_features)
				handle_features.remove(least_feature)
				if max_error < arg_max[k]:
					if k == 2:
						arg_max[k] = max_error
						k += 1
					else:
						stop = False
						while not stop:
							feature_s, error_s = find_least_feature(house_train, iters, features)
							if error_s >= arg_max[k-1]:
								selected_features = copy.deepcopy(handle_features)
								arg_max[k] = max_error
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
	return selected_features, arg_max[len(selected_features)]

if __name__ == '__main__':
	all_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

	print("Downloading the data...")
	price = sklearn.datasets.fetch_california_housing(as_frame=True)
	print("Finish!")

	house = price['frame']
	print(house.info())
	print(house.describe())

	house_train, house_test = seperate_random_pandas(0.2, house)
	# max_features = 4
	# iters = 1000

	# selected_features, error = SFFS(all_features, house_train, max_features, iters)
	# print(selected_features) #['AveBedrms', 'AveRooms', 'MedInc', 'Latitude']


	selected_features = ['AveBedrms', 'AveRooms', 'MedInc', 'Latitude']
	X = trans_xi(house_train, selected_features)
	Y = house_train['MedHouseVal'].values.reshape((house_train['MedHouseVal'].values.size, 1))
	n_iteration = 200

	mu = X.mean(axis=0)
	divid = X.max(axis=0) - X.min(axis=0)

	mu[0] = 0
	divid[0] = 1

	X = standardization(X, mu, divid)

	model = LinearRegression(n_iteration, 0.001, 0.5, 0.0001, 128)
	model.fit(X, Y)

	x_test = trans_xi(house_test, selected_features)
	x_test = standardization(x_test, mu, divid)
	y_test = house_test['MedHouseVal'].values.reshape((house_test['MedHouseVal'].values.size, 1))

	y_pred = model.prediction(x_test)

	error = np.mean(0.5*(y_pred - y_test)**2)
	print(error) #0.3384877315359696
	data = [model.w, model.training_errors, mu, divid]

	with open("./model.pickle", 'wb') as f:
		pickle.dump(data, f)