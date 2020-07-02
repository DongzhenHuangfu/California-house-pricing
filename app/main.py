from fastapi import FastAPI
import numpy as np
from typing import List
import pickle

app = FastAPI()

with open("./model.pickle", 'rb') as f:
	data = pickle.load(f)
w = data[0]

@app.get("/house-pricing/")
async def read_items(features: List[float]):
	input = np.array(features, dtype=np.float16)
	price = input.dot(w)[0]
	return {price}