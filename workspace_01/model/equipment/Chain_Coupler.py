import torch.nn as nn
from compartment import compartment

class Coupler(nn.Module):
    def __init__(self, compartment : compartment, window_size, adj_seq) -> None:
        self.compartment = compartment
        self.window_size = window_size
        self.adj_seq = adj_seq
        
    def Coupler(self):
        pass
    def Set_Adjust_seq(self):
        pass
    def Set_Adjust_count_compartment(self):
        pass
    def Set_DataOrder(self):
        pass