from torch import nn
import torch

##################################
#  Q2
##################################

class ParityLSTM(nn.Module) :

    def __init__(self, hidden_dim=16):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1,hidden_size=hidden_dim,num_layers=1,bias=True,batch_first=True)


    def forward(self, x, x_lens):
        
