import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.decomposition import PCA

np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)

plt.figure()
plt.scatter(X[:,0],X[:,1],edgecolors="k",alpha=0.7)
plt.title("Data Points before PCA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], alpha=0.6)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
origin = pca.mean_

for i, component in enumerate(pca.components_):
    plt.arrow(
        origin[0], origin[1],
        component[0] * 5,
        component[1] * 5,
        color="red" if i == 0 else "blue",
        width=0.03,
        label=f"PC{i+1}"
    )

plt.axis("equal")
plt.show()


plt.figure()
plt.scatter(X_pca[:,0],X_pca[:,1],edgecolors="k",alpha=0.7)
plt.title("Data Points before PCA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

components = pca.components_
print(components)
print (pca.explained_variance_ratio_)

