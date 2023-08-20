import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path,target,train_size):
    data=pd.read_csv(path)
    len=data.shape[0]
    features=[i for i in data.columns if i not in target]
    x1,y1=np.array(data.loc[:len*train_size][features]),np.array(data.loc[:len*train_size]["Admitted"])
    x2,y2=np.array(data.loc[:len*(1-train_size)][features]),np.array(data.loc[:len*(1-train_size)]["Admitted"])
    return x1,x2,y1,y2



def plot_data(x:np.array,y:np.array,xlable:str,ylable:str,lable1:str,lable2:str):
    pos=y==1
    neg=y==0
    plt.scatter(x[pos,0],x[pos,1],c="g",marker="o")
    plt.scatter(x[neg,0],x[neg,1],c="r",marker="x")
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.legend([lable1,lable2])
    plt.show()
    