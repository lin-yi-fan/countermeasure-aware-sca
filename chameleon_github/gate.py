"""Four-condition gate extracted from the research implementation.

Input: preprocessed, training-normalized float tensor [batch, 1000].
Output: logits [batch, 4], ordered BASE, RD, DFS, MRP.
Weights and raw-trace preprocessing are not included.
"""
import torch
from torch import nn

CLASSES = ["BASE", "RD", "DFS", "MRP"]

class Gate(nn.Module):
    """沿用二分类轻量Transformer，输出改为四类。"""
    def __init__(self):
        super().__init__(); self.project=nn.Linear(20,16)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(16,2,32,dropout=.1,batch_first=True),1,enable_nested_tensor=False)
        self.head=nn.Linear(32,4)
    def forward(self,x):
        h=self.encoder(self.project(x.reshape(-1,50,20)))
        return self.head(torch.cat([h.mean(1),h.amax(1)],1))
