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
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Fetch training data and preprocess for modeling
train = pd.read_csv('./data/df_train.csv')

y_train = train[['load_shortfall_3h']]
#train[['Valencia_pressure']] = train[['Valencia_pressure']].fillna(0, inplace=True)
#X_train = train[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed','Valencia_pressure']]
X_train = train.drop(['time','Valencia_wind_deg','Seville_pressure','Valencia_pressure'],axis=1)
X_train.fillna(0,inplace=True)

# Fit model
modelrf2 = RandomForestRegressor(max_depth=4, max_features='sqrt')
print ("Training Model...")
modelrf2.fit(X_train.drop('load_shortfall_3h',axis=1), np.ravel(y_train))

# Pickle model for use within our API
save_path = '../assets/trained-models/modelrf2.pkl'
print (f"Training completed. Saving model to: {save_path}")
pickle.dump(modelrf2, open(save_path,'wb'))
