import pandas as pd
from sklearn.preprocessing import StandardScaler

def detect_fraud(model, X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    y_pred = model.predict(X_scaled)
    return y_pred