import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UZPRFNucrENAFm25csq6eQ/California-housing.csv')
#print (df.head())
#print (df.info())

X = df.drop(columns=['Target'])
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split (X ,y , train_size=0.2, random_state=42)

n = 100
rf = RandomForestRegressor(n_estimators=n, random_state=42)
xgb = XGBRegressor(n_estimators=n, random_state=42)

rf_start = time.time()
rf.fit(X_train, y_train)
rf_end = time.time()
rf_total = rf_end - rf_start

print("Total time taken by Random Forest to train:", rf_total)

xgb_start = time.time()
xgb.fit(X_train, y_train)
xgb_end = time.time()
xgb_total = xgb_end - xgb_start

print("Total time taken by XGBoost to train:", xgb_total)


rf_pred_start = time.time()
y_pred_rf = rf.predict(X_test)
rf_pred_end = time.time()
rf_pred_total = rf_pred_end - rf_pred_start
print ("Total time taken by Random Forest to predict:", rf_pred_total)

xgb_pred_start = time.time()
y_pred_xgb = xgb.predict(X_test)
xgb_pred_end = time.time()
xgb_pred_total = xgb_pred_end - xgb_pred_start

print ("Total time taken by XGBoost to predict:", xgb_pred_total)


mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error (y_test, y_pred_xgb)
r2_rf = r2_score(y_test,y_pred_rf )
r2_xgb = r2_score (y_test, y_pred_xgb)

print ("For Random Forest ->", f"MSE = {mse_rf:.3f} and R^2 = {r2_rf:.3f}" )
print ("For XGBoost ->", f"MSE = {mse_xgb:.3f} and R^2 = {r2_xgb:.3f}" )
