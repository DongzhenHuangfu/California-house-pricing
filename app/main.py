from fastapi import FastAPI
import numpy as np
import pickle

app = FastAPI()

@app.get("/house-pricing/{features}")
def predict(features: float):
	return {features}