#importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#creating data 
x1 = np.concatenate((np.random.normal(10,2,(100,1)),np.random.normal(20,5,(100,1))))
x2 = np.concatenate((np.random.normal(10,2,(100,1)), np.random.normal(30,3,(100,1))))

#visualizing the data
plt.scatter(x1.flatten(),x2.flatten())
plt.show()

#fitting k-means classifier
kmeans = KMeans(n_clusters = 2)
kmeans.fit(np.concatenate((x1,x2),axis = 1))

#visualizing kmeans result
plt.scatter(x1.flatten(),x2.flatten(),c = kmeans.labels_)
plt.show()
