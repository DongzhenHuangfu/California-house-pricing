from fastapi import FastAPI
import numpy as np
from typing import List
import pickle

app = FastAPI()

def trans_xi(data, max_grade=4, show=False):
    ret = np.ones((1, pow(max_grade+1, len(data))), dtype=np.float64)
    for i in range(len(data)):
        for grade in range(max_grade+1):
            if i == 0:
                if grade != 0:
                    ret[:, grade] = data[i] * ret[:, grade-1]
            else:
                now_vec = ret[:, 0:pow(max_grade+1, i)]
                if grade != 0:
                    for j in range(pow(max_grade+1, i)):
                        ret[:, grade * pow(max_grade+1, i) + j] = now_vec[:, j] * pow(data[i], grade)
    return ret


with open("app/model.pickle", 'rb') as f:
	data = pickle.load(f)
	w = data[0]
	mu = data[2]
	divid = data[3]

@app.get("/house-pricing/")
async def read_items(features: List[float]):
	input = np.array(features, dtype=np.float16)

	X = trans_xi(input)
	X_stand = ((X - mu) / divid).astype(np.float16)

	price = X_stand.dot(w)[0, 0]
	return {price}