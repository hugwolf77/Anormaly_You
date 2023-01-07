
import torch.nn as nn
import compartment

class Rnn_compartment(compartment):
    def __init__(self, input_dim, hid_dim, n_layers, dropout) -> None:
        super(Rnn_compartment,self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.LSTM = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input_x):
        x1=self.dropout(input_x)
        outputs, (hidden,cell) = self.LSTM(x1)
        return outputs, hidden, cell