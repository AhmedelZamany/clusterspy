import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('/Users/ahmed/OneDrive/Desktop/Mall_Customers.csv')
df

df.info()

df.isnull().sum()

df.drop('CustomerID',axis=1,inplace=True)


X = df.iloc[:,2:]
X


X.boxplot()


cost_fun = []           
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    print(f'Cost_Function={kmeans.inertia_} with {i} Clusters')
    cost_fun.append(kmeans.inertia_)
    
plt.plot(range(1, 11), cost_fun)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('cost_fun')
plt.show()



model = KMeans(n_clusters=5,n_init=50, random_state=0)
y_kmeans= model.fit_predict(X)



def plotting(y):
    for i in range(5):
        plt.scatter(np.array(X)[y == i,0], np.array(X)[y == i,1], label=f'Cluster {i + 1}')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend(loc = 'best')
    plt.show()
    
    
plotting(y_kmeans)