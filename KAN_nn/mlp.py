import torch
import torch.nn as nn

class MLPPINN(nn.Module):
    def __init__(self, layers, degree):
        super(MLPPINN, self).__init__()
        self.layers = nn.ModuleList()
        # self.all_layers = [2,18,18,18,1]
        
        self.all_layers = [3,50,50,50,50, 50,50,50, 3]

        # self.all_layers = [3,50,50,50,50, 3]
        # self.all_layers = [3,50,50,50,50, 50,3]
        
        # self.all_layers = [3,60,60,3]
        
        for i in range(len(self.all_layers) - 1):
            self.layers.append(nn.Linear(self.all_layers[i], self.all_layers[i + 1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)

