import numpy as np


def costFunction(X,y,Theta):
    m=np.size(X)
    temp=np.square(X*Theta-y)
    J=np.sum(temp)/(2*m)
    return J