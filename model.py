import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('data2.csv')  

X = data.drop('Productivity (%)',axis=1)  
y = data['Productivity (%)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train, y_train)




joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
