from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pickle
from sqlalchemy import create_engine, Column, Integer, Float
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI()
DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

data = load_breast_cancer()
X, Y = data.data, data.target
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X)

class PredictionLog(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(Integer)
    probability = Column(Float)

Base.metadata.create_all(bind=engine)

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "Logistic Regression API"}

@app.post("/predict")
def predict(data: InputData):
    db = SessionLocal()
    x = np.array(data.features).reshape(1, -1)
    pred = model.predict(x)[0]
    prob = model.predict_proba(x)[0][1]
    log = PredictionLog(prediction=int(pred),probability=float(prob))
    db.add(log)
    db.commit()
    db.refresh(log)
    db.close()
    return {
        "class": int(pred),
        "probability": float(prob),
        "log_id": log.id
    }

@app.get("/info")
def info():
    return {
        "dataset length": len(data.data),
        "features": list(data.feature_names),
        "classes": list(data.target_names)
    }

@app.get("/metrics")
def metrics():
    return {
        "accuracy": float(model.score(X, Y)),
        "precision": float(precision_score(Y, y_pred)),
        "recall": float(recall_score(Y, y_pred)),
        "f1-score": float(f1_score(Y, y_pred))
    }

@app.get("/sample")
def sample():
    return {
        "example": X[0].tolist()
    }

@app.get("/logs")
def get_logs():
    db = SessionLocal()
    logs = db.query(PredictionLog).all()
    db.close()
    return logs

@app.delete("/logs/{log_id}")
def delete_log(log_id: int):
    db = SessionLocal()
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if log:
        db.delete(log)
        db.commit()
    db.close()
    return {"status": "deleted"}

@app.delete("/logs")
def delete_all_logs():
    db = SessionLocal()
    db.query(PredictionLog).delete()
    db.commit()
    db.close()
    return {"status": "all logs deleted"}