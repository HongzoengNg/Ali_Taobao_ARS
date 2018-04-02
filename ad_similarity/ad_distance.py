import numpy as np
'''
For this file, it provides function to calculate the similarity between advertisements
Input:
       X --- An sample of advertisement, in formation of array with [adgroup_id,campaign_id,customer,brand]
       Y --- Another sample of advertisement, in formation of array with [adgroup_id,campaign_id,customer,brand]
Output:
       dist ---- A float number, the smaller it is, the higher similarity between 2 advertisements
'''
def get_metric():
    fr=open('metric.txt','r')
    m=[]
    for line in fr.readlines():
        line=list(map(float,line.strip().split()))
        m.append(line)
    m=np.array(m)
    return m

def get_dist(X,Y):
    m=get_metric()
    Z=np.array([X-Y])
    dist=np.sqrt(np.dot(np.dot(Z,m),Z.T))
    return dist[0][0]

