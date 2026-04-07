from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data  = {
    'Hours': [1,2,3,4,5,6,7,8,9],
    'Marks' : [10,20,30,40,50,60,70,80,90]
}
df = pd.DataFrame(data)
X = df[['Hours']]
Y = df['Marks']
model = LinearRegression()
model.fit(X,Y)
pred = model.predict([[9]])

wcss = []
for k in range(1,6):
    Kmeans = KMeans(n_clusters=k,random_state=42)
    Kmeans.fit(X)
    wcss.append(Kmeans.inertia_)
plt.plot(range(1,6),wcss,marker = 'o')
plt.xlabel("clusters")
plt.ylabel("wcss")
plt.title("ELBOW METHOD")
plt.show()

kmeans = KMeans(n_clusters=2,random_state=42)
df['cluster'] = kmeans.fit_predict(X)
print(df)
plt.scatter(9, pred, marker='X', s=200)
plt.scatter(df['Hours'],df['Marks'],c = df['cluster'])
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("Predicted score")
plt.show()
