# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 09:57:23 2021

@author: priyanka kumari
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("pima-indians-diabetes.csv")      


#Question 1
df = df.drop(columns=['class'])

for key in df.keys():
    vals = []
    #Making boxplot to fetch outliers
    _, box = df.boxplot(column=[key], return_type='both') 
    outliers = [flier.get_ydata() for flier in box["fliers"]]
    for i, record in df.iterrows():
        if record[key] in outliers[0]:
            vals.append(i)
    dfd = df.drop(vals)
    for val in vals:
        #Replacing outliers with medians
        df.loc[val , key] = dfd[key].median()                                   

plt.clf()

#Question 1a

min = df.min()
max = df.max()

print("Before Normalisation")
print('Dictionary of Minimum Values')
print(min)
print('Dictionary of Maximum Values')
print(max)

diff = max - min
df1a = (df - min)/diff * 7 + 5

min1a = df1a.min()
max1a = df1a.max()

print("After Normalisation")
print('Dictionary of Minimum Values')
print(min1a)
print('Dictionary of Maximum Values')
print(max1a)


#Question 1b

avg = df.mean()
sd = df.std()

print("Before Standardisation")
print('Mean')
print(avg)
print('Standard Deviation')
print(sd)

df1b = (df - avg)/sd

avg1b = df1b.mean()
sd1b = df1b.std()

print("After Standardisation")
print('Mean')
print(avg1b)
print('Standard Deviation')
print(sd1b)