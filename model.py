import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn_layer import GraphConvolution

class Full_gcn(nn.Module):
    def __init__(self,A_hat, hidden_list,bias=True, dropout=0.5, device='cpu'):
        super(Full_gcn,self).__init__()
        self.A_hat = torch.tensor(A_hat, requires_grad=False, device=device).float()
        self.conv_list = nn.ModuleList([GraphConvolution(in_size, out_size) for in_size, out_size in zip([A_hat.shape[0]]+hidden_list,hidden_list[:-1])])
        # self.drop_list = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(len(self.conv_list))])
        self.fc = nn.Linear(hidden_list[-2], hidden_list[-1])
        
    def forward(self, X):
        for conv in self.conv_list:
            X = F.relu(conv(X, self.A_hat))

        return self.fc(X)

def evaluate(output, labels_e):
    _, labels = output.max(1)
    labels = labels.cpu().numpy()
    return sum([(e-1) for e in labels_e] == labels)/len(labels)