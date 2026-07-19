import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
print (df.info())
print (df.head())

correlation = df.corr()
correlation = abs(correlation['custcat'].sort_values(ascending=False))
print (correlation)

X = df.drop('custcat', axis=1)
y = df['custcat']

# split before normalization to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)


# There is no training because KNN classify based on distance.
model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = model.predict(X_test)

# with small K it it has good acccuracy on training set but bad at generalization.
print ("Train set Accuracy: ", accuracy_score(y_train, model.predict(X_train)))
print ("Test set Accuracy: ", accuracy_score(y_test, y_pred))


K = 10
acc = np.zeros((K))
for n in range(1,K+1):
    #Train Model and Predict  
    model = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = model.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)

def model_accuracy(model_list):
    for i in range(0,K):
        print("Accuracy for k=",i+1,"is",model_list[i])

print ("Accuracy with different number of K:", model_accuracy(acc))
    

