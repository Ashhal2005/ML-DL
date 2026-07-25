import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def showplot (k,method,title):
    plt.plot(k,method,marker="o")
    plt.title(title)
    plt.show()

df = pd.read_csv("/home/ashhal/Desktop/ML-DL/K-mean/Mall_Customers.csv")

#print (df.info())
#print (df.head())

print(df.columns)
df = df [['Annual Income (k$)',
       'Spending Score (1-100)']]

scalar = StandardScaler()
df = scalar.fit_transform(df)

inertia = []
K = range(2,11)
for k in K:
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(df)
    inertia.append(model.inertia_)
showplot (K,inertia,"Elbow Method")

silhouette = []
for k in K:
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    labels = model.fit_predict(df)
    score = silhouette_score(df, labels)
    silhouette.append(score)
showplot(K, silhouette, "silhouette score")


DB_index = []
for k in K:
    model = KMeans(n_clusters=k,random_state=42,n_init=10)
    labels = model.fit_predict(df)
    db_score = davies_bouldin_score(df, labels)
    DB_index.append(db_score)

showplot (K, DB_index, "Davies-Bouldin Index")    