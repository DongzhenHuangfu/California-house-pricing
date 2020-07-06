from fastapi import FastAPI
import numpy as np
from typing import List
import pickle

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

def trans_xi(input, max_grade):
    combines = [1 for i in range(len(input))]
    power_combinations = sum_combines(combines, max_grade)
    # print(power_combinations)
    ret = np.ones(shape=(len(input), len(power_combinations)), dtype=np.float64)
    for i in range(len(power_combinations)):
        for j in range(len(power_combinations[i])):
            ret[:, i] *= pow(input[j], power_combinations[i][j])
    return ret

app = FastAPI()

with open("/app/model.pickle", 'rb') as f:
	data = pickle.load(f)
	w = data[0]
	mus = data[2]
	divids = data[3]

@app.get("/house-pricing/{features}")
async def read_items(features: str):
    max_grade = 4
    split_features = features.split(',')
    if len(split_features) != 2:
        return {"Dimension error: Need 2 features!"}

    float_features = []
    for feature in split_features:
        print(feature)
        float_features.append(float(feature))

    input = np.array(float_features, dtype=np.float16)
    print(input)
    input = (input - mus) / divids + 1
    print(mus)
    print(divids)
    print(input)

    X = trans_xi(input, max_grade)

    price = X.dot(w)[0, 0]
    return {price}