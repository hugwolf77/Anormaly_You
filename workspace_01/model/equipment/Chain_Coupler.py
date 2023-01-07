import torch.nn as nn
from compartment import compartment
from data_provider.data_factory import data_provider

class Coupler(nn.Module):
    def __init__(self, compartment : compartment, window_size, adj_seq) -> None:
        self.compartment = compartment
        self.window_size = window_size
        self.adj_seq = adj_seq
        
    def Coupler(self):
        pass
    def Adjust_seq(self):
        pass
    def Adjust_count_compartment(self):
        pass