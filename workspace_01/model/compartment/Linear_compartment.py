
import torch.nn as nn
import compartment

class Linear_compartment(compartment):
    def __init__(self, input_dim, output_dim, dropout) -> None:
        super(Linear_compartment,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.Linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_x):
        x1 = self.Linear(input_x)
        return x1