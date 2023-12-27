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

print(df.hist(column='income', bins=50))
plt.show()