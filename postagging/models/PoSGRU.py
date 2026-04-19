import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PoSGRU(nn.Module) :

    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True, embed_init=None):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.residual = residual
      
      ##################################
      #  Q5 / Q12
      ##################################
      #TODO
  

    

    def forward(self, x):
      ##################################
      #  Q5
      ##################################
      #TODO
      return out