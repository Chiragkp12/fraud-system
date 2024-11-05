import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Convert Time to datetime
    data['Time'] = pd.to_datetime(data['Time'])
    data['Hour'] = data['Time'].dt.hour

  
  data.drop(['Time'], axis=1, inplace=True)

   # Split the data into features and target
   X = data.drop(['IsFraud'], axis=1)
   y = data['IsFraud']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Scale the features
  scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
