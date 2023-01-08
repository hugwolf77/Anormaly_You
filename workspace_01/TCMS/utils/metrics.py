import numpy as np
from numpy import dot
from numpy.linalg import norm

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

# DLinear 
def CORR(pred, true):
    u = (abs(true - true.mean(0)) * abs(pred - pred.mean(0))).sum(0)
    A = np.sqrt(((true - true.mean(0)) ** 2).sum(0))
    B = np.sqrt(((pred - pred.mean(0)) ** 2).sum(0))
    d = A*B
    d += 1e-12
    return (u / d).mean()

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return mae,mse,rmse,mape,mspe,rse,corr
