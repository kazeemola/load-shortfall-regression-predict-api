"""
    Simple file to create a sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
train[['Valencia_pressure']] = train[['Valencia_pressure']].fillna(0)
X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed','Valencia_pressure']]

# Fit model
model3 = LinearRegression(normalize=True)
print ("Training Model...")
model3.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/model3.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(model3, open(save_path,'wb'))
