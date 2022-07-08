import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df=pd.DataFrame(pd.read_csv('pima-indians-diabetes.csv'))

#dropping the class column
df1 = df.drop(['class'] ,axis = 1)
col=[]
for i in df.columns:
    if(i!="class"):
        col.append(i)

#Replacing the outliers with median.
for i in df1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    
    Median = df1[i].median()
    df1[i].values[df1[i] > (Q3 + (1.5 * IQR))] = Median
    df1[i].values[df1[i] < (Q1 - (1.5 * IQR))] = Median
    
# Q1. B Data is taken.
df2 = df1.copy()
#Standardizing using mean and standard deviation
for i in df2:
    df2[i] = (df2[i] - df2[i].mean()) / (df1[i].std())    

#Q3 A    
pca_ = PCA(n_components = 2)
data=pd.DataFrame(data = pca_.fit_transform(df2),columns = ['PC1', 'PC2'])
print(round(data.cov(),3))
print(pca_.explained_variance_ratio_)
plt.scatter(data['PC1'],data['PC2'])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Plot of data after dimensionality reduction:")
plt.show()


#Q3 B
covmatrix=df1.corr()
eigenvalues,eigenvectors=np.linalg.eig(covmatrix.to_numpy())
eigenvalues[::-1].sort()
print('\n')
print("Eigenvalues in descending order:")
print(eigenvalues)
print("\n")
plt.bar(np.arange(1,9),eigenvalues)
plt.xlabel("Eigenvalues")
plt.ylabel("Magnitude of Eigenvalues")
plt.title("Plot of Eigenvalues in descending order:")
plt.show()

#Q3 C
rec_error={}
for i in range(1,9):
    pca= PCA(n_components=i)
    X=pca.fit_transform(df2)
    rec_x=pca.inverse_transform(X)
    z1=pd.DataFrame(data = X)
    rec_d=pd.DataFrame(data = rec_x,columns = col)
    rec_error[i]=np.linalg.norm(rec_d-df2,None)
    print(round(z1.cov(),3))
    round(z1.cov(),3).to_excel("haalo1.xls")

#plotting the errors
plt.plot(rec_error.keys(),rec_error.values())
plt.xlabel("L")
plt.ylabel("Recontruction error")
plt.title("Line plot to demonstrate reconstruction error vs. components:")
plt.show()
 
#Q3 D  
#Comparing the two matrices.
print("\n")
print("Comparing these two:")
print("Covariance Matrix of Original data")  
print(round(df2.cov(),3))

pca= PCA(n_components=8)
x=pca.fit_transform(df2)
d=pd.DataFrame(data = x)
print("Covariance matrix of 8 dimensional representation:")
print(round(d.cov(),3))

