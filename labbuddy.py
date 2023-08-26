import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path:str,target:str,train_size:float,neglect_feature:list):
    data=pd.read_csv(path)
    len=data.shape[0]
    features=[i for i in data.columns if i not in target if i not in neglect_feature ]
    x1,y1=np.array(data.loc[:len*train_size][features]),np.array(data.loc[:len*train_size]["Admitted"])
    x2,y2=np.array(data.loc[:len*(1-train_size)][features]),np.array(data.loc[:len*(1-train_size)]["Admitted"])
    return x1,x2,y1,y2



def plot_data(x:np.array,y:np.array,xlable:str,ylable:str,lable1:str,lable2:str,show:bool):
    pos=y==1
    neg=y==0
    plt.scatter(x[pos,0],x[pos,1],c="g",marker="o")
    plt.scatter(x[neg,0],x[neg,1],c="r",marker="x")
    if show==True:
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.legend([lable1,lable2])
        plt.show()


def sigmoid(z):
    val=1+np.exp(-z)
    g=1/val
    return g

def map_feature(x1, x2):
    x1 = np.atleast_1d(x1)
    x2 = np.atleast_1d(x2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((x1**(i-j) * (x2**j)))
    return np.stack(out, axis=1)


def plot_decision_boundary(w, b, x, y,xlable:str,ylable:str,lable1:str,lable2:str,show:bool):
    plot_data(x[:, 0:2], y,xlable,ylable,lable1,lable2,False)
    if x.shape[1] <= 2:
        plot_x = np.array([min(x[:, 0]), max(x[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        plt.plot(plot_x, plot_y, c="b")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)     
        z = np.transpose(z)
        plt.contour(u,v,z, levels = [0.5], colors="g")
    if show==True:
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.legend(["Decision boundary",lable2,lable1,])
        plt.show()