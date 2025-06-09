import torch
import torch.nn as nn
import torch.nn.functional as F
from xLSTM.utils import BlockDiagonal, CausalConv1D

class mLSTMblock(nn.Module):
    def __init__(self, x_example, factor, depth, dropout=0.2):
        super().__init__()
        self.input_size = x_example.shape[2]
        self.hidden_size = int(self.input_size*factor)
        
        self.ln = nn.LayerNorm(self.input_size)
        
        self.left = nn.Linear(self.input_size, self.hidden_size)
        self.right = nn.Linear(self.input_size, self.hidden_size)
        
        self.conv = CausalConv1D(self.hidden_size, self.hidden_size, int(self.input_size/10)) 
        self.drop = nn.Dropout(dropout+0.1)
        
        self.lskip = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.wq = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wk = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.wv = BlockDiagonal(self.hidden_size, self.hidden_size, depth)
        self.dropq = nn.Dropout(dropout/2)
        self.dropk = nn.Dropout(dropout/2)
        self.dropv = nn.Dropout(dropout/2)
        
        self.i_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.f_gate = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_gate = nn.Linear(self.hidden_size, self.hidden_size)

        self.ln_c = nn.LayerNorm(self.hidden_size)
        self.ln_n = nn.LayerNorm(self.hidden_size)
        
        self.lnf = nn.LayerNorm(self.hidden_size)
        self.lno = nn.LayerNorm(self.hidden_size)
        self.lni = nn.LayerNorm(self.hidden_size)
        
        self.GN = nn.LayerNorm(self.hidden_size)
        self.ln_out = nn.LayerNorm(self.hidden_size)

        self.drop2 = nn.Dropout(dropout)
        
        self.proj = nn.Linear(self.hidden_size, self.input_size)
        self.ln_proj = nn.LayerNorm(self.input_size)
        
        self.init_states(x_example)
    
    def forward(self, x):
    assert x.ndim == 3
    batch_size = x.size(0)

    x = self.ln(x)
    
    left = self.left(x)
    right = F.silu(self.right(x))
    
    left_left = left.transpose(1, 2)
    left_left = F.silu(self.drop(self.conv(left_left).transpose(1, 2)))
    l_skip = self.lskip(left_left)
    
    q = self.dropq(self.wq(left_left))
    k = self.dropk(self.wk(left_left))
    v = self.dropv(self.wv(left))
    
    i = torch.exp(self.lni(self.i_gate(left_left)))
    f = torch.exp(self.lnf(self.f_gate(left_left)))
    o = torch.sigmoid(self.lno(self.o_gate(left_left)))

    # 状态扩展到 batch 维度
    ct_1 = self.ct_1.repeat(batch_size, 1, 1)
    nt_1 = self.nt_1.repeat(batch_size, 1, 1)

    ct = f * ct_1 + i * v * k
    ct = torch.mean(self.ln_c(ct), dim=1, keepdim=True)
    self.ct_1 = torch.mean(ct, dim=0, keepdim=True).detach()

    nt = f * nt_1 + i * k
    nt = torch.mean(self.ln_n(nt), dim=1, keepdim=True)
    self.nt_1 = torch.mean(nt, dim=0, keepdim=True).detach()

    ht = o * ((ct * q) / torch.max(nt * q))
    ht = ht

    left = self.drop2(self.GN(ht + l_skip))

    out = self.ln_out(left * right)
    out = self.ln_proj(self.proj(out))

    return out
