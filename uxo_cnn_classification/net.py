import torch
from torch import nn
from torch.nn import functional

class ConvNet(nn.Module):
    
    def __init__(self, network_geometry_3d, network_geometry_2d, input_width, input_height, input_depth, n_class):
        super().__init__()
        self.netgeometry3d=network_geometry_3d
        self.netgeometry2d=network_geometry_2d
        #self.nt = len(network_geometry)
        self.input_width = input_width
        self.input_height = input_height
        self.input_depth = input_depth
        self.n_class = n_class
        #self.batch_norm = nn.BatchNorm2d(network_geometry[0])
        self.init_params()
        
        
    def init_params(self):
        initial_scaling = 1e-2
        nlayers3d = len(self.netgeometry3d)
        self.K3d = nn.ParameterList([]) #[]
        self.bnorms3d = nn.ModuleList([nn.BatchNorm3d(self.netgeometry3d[0])])
        self.bias = None #[]
        for i in range(nlayers3d-1):
            n_in = self.netgeometry3d[i]
            n_out = self.netgeometry3d[i+1]
            Ki = torch.Tensor(n_out, n_in, 3, 3, 3)
            #nn.init.kaiming_uniform_(Ki)
            Ki.data = torch.randn(n_out, n_in, 3, 3, 3)  * initial_scaling
            self.K3d.append(Ki)
            self.bnorms3d.append(nn.BatchNorm3d(n_out))
        
        nlayers2d = len(self.netgeometry2d)
        self.K2d = nn.ParameterList([]) #[]
        self.bnorms2d = nn.ModuleList([nn.BatchNorm2d(self.input_depth*self.netgeometry3d[-1])])
        n_in = n_out*self.input_depth
        for i in range(nlayers2d-1):
            #n_in = self.netgeometry2d[i]
            n_out = self.netgeometry2d[i]
            Ki = torch.Tensor(n_out, n_in, 3, 3)
            #nn.init.kaiming_uniform_(Ki)
            Ki.data = torch.randn(n_out, n_in, 3, 3)  * initial_scaling
            self.K2d.append(Ki)
            self.bnorms2d.append(nn.BatchNorm2d(n_out))
            n_in = n_out
        #self.Kout = nn.Parameter(torch.Tensor(n_class, n_out*27, 3, 3))
        self.Kout = nn.Parameter(torch.Tensor(self.n_class, n_in, 1, 1))
        #nn.init.kaiming_uniform_(Ki)
        self.Kout.data = torch.randn(self.n_class, n_in, 1, 1)  * initial_scaling
        self.bnout = nn.BatchNorm2d(self.n_class)
        self.biasout = nn.Parameter((torch.randn(self.n_class)*initial_scaling))
    
    def forward(self, X): 
        X = self.bnorms3d[0](X)
        if self.bias is None:
            self.bias = [None]*len(self.K3d)
        for i, Ki, bn, b in zip(range(len(self.netgeometry3d)), self.K3d, self.bnorms3d[1:], self.bias):
            z = functional.conv3d(X, Ki, stride=1, padding=1, bias=b)
            z = bn(z)
            z = functional.leaky_relu(z,negative_slope=0.2)
            z = functional.max_pool3d(z, 3, stride=1, padding=1)
            X = z
        X = X.transpose(3,4).transpose(2,3).reshape(X.shape[0],self.netgeometry3d[-1]*self.input_depth,self.input_height,self.input_width)
        X = self.bnorms2d[0](X)
        if self.bias is None:
            self.bias = [None]*len(self.K2d)
        for i, Ki, bn, b in zip(range(len(self.netgeometry2d)), self.K2d, self.bnorms2d[1:], self.bias):
            z = functional.conv2d(X, Ki, stride=1, padding=1, bias=b)
            z = bn(z)
            z = functional.leaky_relu(z,negative_slope=0.2)
            z = functional.max_pool2d(z, 3, stride=1, padding=1)
            X = z
        X = functional.conv2d(X, self.Kout, stride=1, padding=0, bias=self.biasout) # change padding when needed!!
        S = self.bnout(X)
        #S = functional.relu(X)
        #S = torch.matmul(X, self.W) + self.fc_bias.unsqueeze(0)
        probs = functional.softmax(S, dim=1)
        return S, probs