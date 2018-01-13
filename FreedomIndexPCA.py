'''
All data are index values for 2014

Sources
-------
Fraser Institute's Economic Freedom of the World Index: https://www.fraserinstitute.org/studies/economic-freedom-of-the-world-2017-annual-report*
Cato Institute's Personal Freedom Index: https://www.cato.org/human-freedom-index
Social Progress Imperative's Social Progress Index: https://www.socialprogressindex.com*

*original data were divided by 10 so that all three indices had values between 0 and 10
'''
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from scipy.special import erfinv

#Define filepaths
base_path = os.path.split(__file__)[0]
data_file = os.path.join(base_path, 'IndexData.csv')
corr_file = os.path.join(base_path, 'IndexCorrelations.csv')

#Read in data
data = pd.read_csv(data_file).dropna()

X = (data[['Economic', 'Personal', 'Social']].copy()) #Create a copy of the data frame to identify principal components
X -= X.mean(0) #Subtract each index from its mean value
n = X.shape[0] #Number of countries
(U, S, VT) = np.linalg.svd(X.T/np.sqrt(n-1)) #Apply singular-value decomposition to identify principal components

projection = np.dot(U.T, X.T).T #Project data onto principal component basis
data['Combined'] = -projection[:, 0] #The combined index is the strongest principal component, which is listed first
data['Combined'] += (5 - data['Combined'].mean()) #Add a factor so the mean value of the combined index is 5

#Write data frames to file
data.set_index('Countries (2014)').sort_values('Combined', ascending = False).to_csv(data_file)
data.corr().to_csv(corr_file)