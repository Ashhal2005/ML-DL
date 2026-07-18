
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv"
df = pd.read_csv(path)

# Features and Target
X = df.drop(["Drug"], axis=1)
y = df["Drug"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)



# Select categorical columns
cat_cols = X_train.select_dtypes(include="object").columns



# Encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")


X_train_values = encoder.fit_transform(X_train[cat_cols])
X_test_values = encoder.transform(X_test[cat_cols])


# Combine numerical and encoded categorical columns
X_train = pd.concat([X_train.drop(columns=cat_cols), pd.DataFrame(X_train_values,
 columns=encoder.get_feature_names_out(cat_cols), index=X_train.index)], axis=1)

X_test = pd.concat([X_test.drop(columns=cat_cols), pd.DataFrame(X_test_values,
 columns=encoder.get_feature_names_out(cat_cols), index=X_test.index)], axis=1)



label_encoder = LabelEncoder()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Training the Tree

model = DecisionTreeClassifier( criterion="entropy", max_depth=4, random_state=42)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print("Accuracy of Decision Tree Classifier with test data:", accuracy_score(y_test, y_pred))
print("Accuracy of Decision Tree Classifier with train data:",accuracy_score(y_train, train_pred))

print ("Classification Report ON test data:", classification_report(y_test, y_pred))
print("Classification Report for Train Data:", classification_report(y_train, train_pred))
