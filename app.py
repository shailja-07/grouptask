from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

class Data(BaseModel):
    Age: int
    Projects_Completed: int
    Satisfaction_Rate: float  
    Feedback_Score: float
    Salary: float

@app.get("/")
def read_root():
    return {"message": "Welcome!"}

@app.get('/{name}')
def get_name(name: str):
    return {'lets know your productivity': f'{name}'}


@app.post("/predict")
def predict(data: Data):
    input_data = np.array([[

        data.Age, 
        data.Projects_Completed,
        data.Satisfaction_Rate,
        data.Feedback_Score,   
        data.Salary
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    return {"prediction": int(prediction[0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
