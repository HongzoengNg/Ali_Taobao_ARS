import pandas as pd
import numpy as np
import random
from metric_learn import ITML_Supervised

def get_metric():
    ad=pd.read_csv('ad_feature.csv',header=0)
    data=ad.values
    m=np.array([[.0,.0,.0,.0],[.0,.0,.0,.0],[.0,.0,.0,.0],[.0,.0,.0,.0]])
    itml = ITML_Supervised(num_constraints=200)
    for i in range(100):
        data_r=np.array(random.sample(data.tolist(),int(len(data)/100)))
        x=data_r[:,[0,2,3,4]]
        y=data_r[:,1]
        itml.fit(x,y)
        m=m+itml.metric()
    m=m/100
    return m

def output2file(metric):
    fw=open('metric.txt','w')
    for i in range(len(metric)):
        string=str(metric[i][0])
        for j in range(1,len(metric[i])):
            string=string+'\t'+str(metric[i][j])
        fw.write(string+'\n')
    fw.close()
    return

if __name__ == '__main__':
    m=get_metric()
    output2file(m)
