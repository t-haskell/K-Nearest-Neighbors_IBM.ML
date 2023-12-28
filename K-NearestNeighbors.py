'''
Lab by -> Thomas Haskell

Topic -> Classification with K-Nearest Neighbors
Source -> IBM Machine Learning with Python Certification Course

Objectives:
1. Use K-Nearest Neighbors to classify a customer data set
2. Understand how data points with similar features can help classify
unkown observations
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# With this data set, we will classify telecommunication customers into similar use groups 
# based on demographics like region, age, age, and marital status. 
# Target Field: "custcat" = number value of an customer identification group

# There are 4 possible customer identification groups: 1- Basic Service, 2- E-Service, 
# 3- Plus Service, and 4- Total Service. 

import urllib.request
import ssl

# workaround for SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
def download(url, filename):
    urllib.request.urlretrieve(url, filename)
    print("Download Complete")
    
path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"

download(path, "teleCust1000t.csv") #(might take a minute to download on first run)#
# reading in data to a pandas dataframe
df = pd.read_csv("teleCust1000t.csv")
print(df.head())

print(df['custcat'].value_counts())

df.hist(column='income', bins=50)
plt.show()

## Creating Feature Sets
print(df.columns)
# converting to a Numpy array to be compatible with scikit-learn
X = df[['region', 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']]
# finding labels
y = df['custcat'].values
print(y[0:5])

## Splitting Test & Train Datasets
# To opitmize out-of-sample accuracy, we split the data into mutually exclusive
# sets for training and testing.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# Visualizing the training set
print(f"Train set: {X_train.shape} {y_train.shape}")
# Visualing the testing set
print(f"Test set: {X_test.shape} {y_test.shape}")

## Normalizing data (good practice to apply zero mean and unit variance when finding distance)
X_train_norm = preprocessing.StandardScaler().fit(X_train).transform(X_train.astype(float))
print(X_train_norm[0:5])


######## Implementing K-Nearest Neighbors method ########
from sklearn.neighbors import KNeighborsClassifier
# Training 
k = 4 # number of neighbors to be considered
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train_norm, y_train)
print(neigh)

# Predicting
X_test_norm = preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))
X_test_norm[0:5]
yhat = neigh.predict(X_test_norm)

## Accuracy Evaluation
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train_norm)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


## New model with 6 neighbors instead of 4
sleigh = KNeighborsClassifier(n_neighbors = 6).fit(X_train_norm,y_train)
yhat6 = sleigh.predict(X_test_norm)
print("Train set Accuracy (using 6 neighbors): ", metrics.accuracy_score(y_train, sleigh.predict(X_train_norm)))
print("Test set Accuracy (using 6 neighbors): ", metrics.accuracy_score(y_test, yhat6))


######## Finding Optimal Number of Neighbors ########
# Process -> test multiple values of k starting at 1, and see which results in highest accuracy.

Ks = 10
# creating arrays to hold metrics of different k-value models
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train model with n neighbors and predict
    nneigh = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm, y_train)
    yhat = nneigh.predict(X_test_norm)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat) # recording mean each model
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0]) # recording std. deviation for each model

mean_acc


#### Visualizing the results ####
plt.plot(range(1,Ks), mean_acc, 'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha = 0.10)
plt.fill_between(range(1,Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
