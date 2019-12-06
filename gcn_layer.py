import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    def __init__(self,in_size, out_size, bias=True, device='cpu'):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size; self.out_size = out_size
        self.kernel = nn.parameter.Parameter(torch.FloatTensor(in_size, out_size).to(device))
        var = 2./(self.kernel.size(1)+self.kernel.size(0))
        self.kernel.data.normal_(0,var)
        if bias:
            self.bias=nn.parameter.Parameter(torch.FloatTensor(out_size).to(device))
            self.bias.data.normal_(0,var)
        else:
            self.register_parameter("bias", None)
        
    def forward(self, X, adj):
        X = torch.mm(X,self.kernel)
        X = torch.mm(adj, X)
        if self.bias is not None:
            return X + self.bias
        else:
            return X
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'