import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
df = pd.read_csv(file_path)
#print(df.head())

#print (df.info())
#print (df.describe())

X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

#Spliting the dataset into training and testing sets to prevent data-leakage 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)


#Scaling the continuous features
float_columns = X_train.select_dtypes(include=["float64"]).columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[float_columns])
X_test_scaled = scaler.transform(X_test[float_columns])
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=float_columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=float_columns, index=X_test.index)
X_train = pd.concat([X_train_scaled_df, X_train.drop(float_columns, axis=1)], axis=1)
X_test = pd.concat([X_test_scaled_df, X_test.drop(float_columns, axis=1)], axis=1)

#Encoding categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first',handle_unknown= "ignore")
str_columns = X_train.select_dtypes(include=["object"]).columns
    
X_train_encoded = encoder.fit_transform(X_train[str_columns])
X_test_encoded = encoder.transform(X_test[str_columns])
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(str_columns), index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(str_columns), index=X_test.index)

X_train = pd.concat ([X_train.drop(str_columns, axis=1), X_train_encoded_df], axis=1)
X_test = pd.concat ([X_test.drop(str_columns, axis=1), X_test_encoded_df], axis=1)

# Target feature encoding
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Training the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)


# Evaluating the model
print ("Confusion Matrix:", confusion_matrix(y_test, y_pred))
print ("Accuracy Score: ", accuracy_score(y_test, y_pred))
print ("Classification Report:", classification_report(y_test, y_pred))
