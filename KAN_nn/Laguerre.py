import torch
import torch.nn as nn
import numpy as np

class LaguerreKANLayer(nn.Module):
    """
    表示基于拉盖尔多项式的KAN网络的单层。
    """
    def __init__(self, input_dim, output_dim, degree, alpha):
        """
        初始化层参数。

        Args:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            degree (int): 拉盖尔多项式的阶数
            alpha (float): 广义拉盖尔多项式的参数
        """
        super(LaguerreKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha

        # 初始化拉盖尔多项式系数
        self.laguerre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.laguerre_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        """
        前向传播，使用显式展开的拉盖尔多项式。

        Args:
            x (torch.Tensor): 输入张量 (batch_size, input_dim)
        Returns:
            torch.Tensor: 输出张量 (batch_size, output_dim)
        """
        x = x.view(-1, self.input_dim)
        x = torch.tanh(x)  # 归一化到[-1, 1]

        # 初始化拉盖尔多项式张量
        laguerre = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        
        # 显式写出前几阶广义拉盖尔多项式 L_n^(α)(x)
        if self.degree >= 0:
            laguerre[:, :, 0] = 1.0  # L_0^α(x) = 1
            
        if self.degree >= 1:
            laguerre[:, :, 1] = 1 + self.alpha - x  # L_1^α(x) = 1 + α - x
            
        if self.degree >= 2:
            laguerre[:, :, 2] = (
                (self.alpha + 2) * (self.alpha + 1) / 2 
                - (self.alpha + 2) * x 
                + x**2 / 2
            )  # L_2^α(x) = [(α+2)(α+1) - 2(α+2)x + x^2]/2
            
        if self.degree >= 3:
            laguerre[:, :, 3] = (
                (self.alpha + 3) * (self.alpha + 2) * (self.alpha + 1) / 6 
                - (self.alpha + 3) * (self.alpha + 2) * x / 2 
                + (self.alpha + 3) * x**2 / 2 
                - x**3 / 6
            )  # L_3^α(x)
            
        if self.degree >= 4:
            laguerre[:, :, 4] = (
                (self.alpha + 4) * (self.alpha + 3) * (self.alpha + 2) * (self.alpha + 1) / 24 
                - (self.alpha + 4) * (self.alpha + 3) * (self.alpha + 2) * x / 6
                + (self.alpha + 4) * (self.alpha + 3) * x**2 / 4 
                - (self.alpha + 4) * x**3 / 6 
                + x**4 / 24
            )  # L_4^α(x)

        # 使用einsum进行拉盖尔插值
        y = torch.einsum('bid,iod->bo', laguerre, self.laguerre_coeffs)
        y = y.view(-1, self.output_dim)
        return y

class LaguerreKAN(nn.Module):
    """
    表示基于拉盖尔多项式的Kolmogorov-Arnold网络(KAN)。
    """
    
    def __init__(self, network, degree, alpha=0.0):
        """
        初始化LaguerreKAN网络。

        Args:
            network (list[int]): 定义每层节点数的列表
            degree (int): 拉盖尔多项式的阶数
            alpha (float): 广义拉盖尔多项式的参数
        """
        super(LaguerreKAN, self).__init__()
        self.network = network
        self.layers = nn.ModuleList()

        # 根据指定的网络架构定义层
        for i in range(len(network) - 1):
            self.layers.append(LaguerreKANLayer(network[i], network[i + 1], degree, alpha))

    def forward(self, x):
        """
        网络的前向传播。

        Args:
            x (torch.Tensor): 输入张量 (batch_size, input_dim)
        Returns:
            torch.Tensor: 输出张量 (batch_size, output_dim)
        """
        x = x.view(-1, self.network[0])  # 将输入重塑以匹配第一层

        # 顺序通过所有层
        for layer in self.layers:
            x = layer(x)
        return x

class LaguerrePINN(nn.Module):
    """
    表示使用拉盖尔多项式基的物理信息神经网络(PINN)。
    """

    def __init__(self, network, degree, alpha=0.0):
        """
        初始化LaguerrePINN模型。

        Args:
            network (list[int]): 定义每层节点数的列表
            degree (int): 拉盖尔多项式的阶数
            alpha (float): 广义拉盖尔多项式的参数
        """
        super(LaguerrePINN, self).__init__()
        self.model = LaguerreKAN(network, degree, alpha)  # 定义核心LaguerreKAN网络
        self.network = network
        self.layers = nn.ModuleList()

    def forward(self, x, x_min=0, x_max=1):
        """
        LaguerrePINN模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量 (batch_size, input_dim)
            x_min (float, optional): 归一化的最小值 (默认: 0)
            x_max (float, optional): 归一化的最大值 (默认: 1)
        Returns:
            torch.Tensor: 输出张量 (batch_size, output_dim)
        """
        return self.model(x)