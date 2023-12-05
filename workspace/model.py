import torch
import torch.nn as nn


class BertFC(nn.Module):
  def __init__(self,embedding_dim,output_size,dropout_rate):
    super(BertFC,self).__init__()
    self.fc = nn.Linear(embedding_dim,7448)
    self.fc1 = nn.Linear(7448,1862)
    self.fc2 = nn.Linear(1862,output_size)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self,x):
    out = self.dropout(x)
    out = self.fc(torch.tanh(out))
    out = self.fc1(out)
    out = self.dropout(out)
    out = self.fc2(out)
    return out
