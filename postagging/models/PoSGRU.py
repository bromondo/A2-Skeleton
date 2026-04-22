import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PoSGRU(nn.Module) :

    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True, embed_init=None):
      super().__init__()
      self.hidden_dim = hidden_dim
      self.residual = residual
      self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=1)
      self.linear = nn.Linear(in_features=embed_dim,out_features=hidden_dim)
      self.GRU = nn.GRU(input_size=hidden_dim,hidden_size=hidden_dim//2,num_layers=num_layers,bidirectional=True,batch_first=True)
      self.classify = nn.Sequential(
         nn.Linear(hidden_dim, hidden_dim),
         nn.GELU(),
         nn.Linear(hidden_dim, output_dim)
      )
      ##################################
      #  Q5 / Q12
      ##################################
  

    

    def forward(self, x):
      ##################################
      #  Q5
      ##################################
      
      em = self.embed(x)
      ln = self.linear(em)
      g, _= self.GRU(ln)
      if self.residual:
         g = g + ln
      out = self.classify(g)
      return out