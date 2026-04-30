from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pickle

data = load_breast_cancer()
X, Y = data.data, data.target

model = LogisticRegression(max_iter=10000)
model.fit(X, Y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
