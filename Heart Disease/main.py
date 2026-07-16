import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

col= [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

df = pd.read_csv('/home/ashhal/Desktop/ML-DL/Heart Disease/data/processed.cleveland.data', names=col , na_values='?')  

#df.describe()

print("% of missing values:", df.isnull().sum().sum()/len(df)*100)

# 2 columns has missing values, we will drop them coz they are in less percentage.
df = df.dropna()   
print (df.isnull().sum())

X= df.drop('target', axis=1)
y= df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

#accuracy with StandardScalar 0.62
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.2f}")

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))




