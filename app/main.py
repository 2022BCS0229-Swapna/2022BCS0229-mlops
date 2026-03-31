from fastapi import FastAPI
import joblib

ROLL_NO = "2022BCS0229"
NAME = "Swapna"

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def health():
    return {
        "Name": NAME,
        "Roll No": ROLL_NO
    }

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data]).tolist()
    return {
        "prediction": prediction,
        "Name": NAME,
        "Roll No": ROLL_NO
    }