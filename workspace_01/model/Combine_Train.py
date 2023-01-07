import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from compartment import compartment, Rnn_compartment, Linear_compartment
from equipment import Decompose, RInsNorm, Chain_Coupler


class Model(nn.Module):
    '''
    TS Train Compartmnet 
    '''
    def __init__(self,configs) -> None:
        super(Model,self).__init__()
        
        self.window_size = configs.window_size
        self.adj_seq = configs.adj_seq
        
        self.batch_size = configs.batch_size
        self.input_seq = configs.input_seq
        self.pred_seq = configs.pred_seq
        
        self.input_dim = configs.input_dim
        self.output_dim = configs.output_dim
        
        self.hid_dim = configs.hid_dim
        self.dropout = configs.dropout
        
        self.kernel_size = configs.moving_avg
        
        if isinstance(self.kernel_size, list):
            self.decomposition = Decompose.series_decomp_multi(self.kernel_size)
        else:
            self.decomposition = Decompose.series_decomp(self.kernel_size)
            
        self.RIN = RInsNorm()
            
        
        self.Linear_compartment = Linear_compartment(self.input_dim, self.output_dim, self.dropout)
        self.LSTM_compartmnet = Rnn_compartment(self.input_dim, self.hid_dim, self.dropout)
        
        self.Chain_Coupler = Chain_Coupler(compartment,self.window_size,self.adj_seq)
        
    def forward(self,input_x):
        freq,trend = self.decomposition(input_x)
        trend = self.RIN.Activate_InsNorm(trend)
        x_trend = self.Linear_compartment(trend)
        x_trend = self.RIN.Deactivate_InsNorm(x_trend)
        x_freq = self.LSTM_compartmnet(freq)
        res = input_x - (x_trend + x_freq)
        out = x_trend + x_freq + res
        return out