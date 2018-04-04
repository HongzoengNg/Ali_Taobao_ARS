import numpy as np
import pandas as pd
'''
For this file, it provides function to calculate the similarity between advertisements
Input:
       X --- A dataframe of advertisements,in formation of columns with [adgroup_id,campaign_id,customer,brand]
       Y --- Another dataframe of advertisements, in formation of columns with [adgroup_id,campaign_id,customer,brand]
Output:
       dist ---- A series of float number
                 for each of the number, the smaller it is, the higher similarity between 2 advertisements
'''
def get_metric():
    fr=open('metric.txt','r')
    m=[]
    for line in fr.readlines():
        line=list(map(float,line.strip().split()))
        m.append(line)
    m=np.array(m)
    return m

def get_dist_single(x,y,M):
    Z=np.array([x-y])
    dist=np.sqrt(np.dot(np.dot(Z,M),Z.T))
    return dist[0][0]

def get_dist_df(X,Y):
    M=get_metric()
    dist_lst=[]
    for i in range(len(X)):
        x=np.array(X.iloc[i])
        y=np.array(Y.iloc[i])
        dist_lst.append(get_dist_single(x,y,M))
    dist=pd.DataFrame({'distance':dist_lst})
    return dist
