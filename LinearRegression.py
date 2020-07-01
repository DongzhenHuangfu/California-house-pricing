import sklearn.datasets
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import multiprocessing as mp

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
            print('iter: ',error)
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

def train_features(features, house_train, n_iteration):
	X = trans_xi(house_train, features)
	Y = house_train['MedHouseVal'].values.reshape((house_train['MedHouseVal'].values.size, 1))

	mu = X.mean(axis=0)
	divid = X.max(axis=0) - X.min(axis=0)

	mu[0] = 0
	divid[0] = 1

	X_Stand = standardization(X, mu, divid)

	x_train, x_valid, y_train, y_valid = seperate_random_np(0.2, X_Stand, Y)

	model = LinearRegression(n_iteration, 0.001, 0.5, 0.0001, 128)
	model.fit(x_train, y_train)

	error = np.mean(0.5 * (model.prediction(x_valid) - y_valid) ** 2)
	return model.w, error, mu, divid

if __name__ == '__main__':
	all_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

	print("Downloading the data...")
	price = sklearn.datasets.fetch_california_housing(as_frame=True)
	print("Finish!")

	house = price['frame']
	print(house.info())
	print(house.describe())

	house_train, house_test = seperate_random_pandas(0.2, house)

	features = ['Latitude', 'Longitude']
	w, error, mu, divid = train_features(features, house_train, 1000)