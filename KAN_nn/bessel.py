import torch
import torch.nn as nn
import numpy as np

class BesselKANLayer(nn.Module):
    """
    表示基于贝塞尔多项式的KAN网络的单层。
    """
    
    def __init__(self, input_dim, output_dim, degree):
        """
        初始化层的输入/输出维度和贝塞尔多项式阶数。

        Args:
            input_dim (int): 输入特征数。
            output_dim (int): 输出特征数。
            degree (int): 使用的贝塞尔多项式阶数。
        """
        super(BesselKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # 初始化贝塞尔多项式系数
        self.bessel_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.bessel_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        """
        使用贝塞尔多项式的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, input_dim)。

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, output_dim)。
        """
        x = x.view(-1, self.input_dim)
        x = torch.tanh(x)  # 归一化到[-1, 1]

        # 初始化贝塞尔多项式张量
        bessel = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        
        # 显式写出前几阶贝塞尔多项式
        if self.degree >= 1:
            bessel[:, :, 1] = x + 1  # y1(x) = x + 1
        if self.degree >= 2:
            bessel[:, :, 2] = 3*x**2 + 3*x + 1  # y2(x) = 3x^2 + 3x + 1
        if self.degree >= 3:
            bessel[:, :, 3] = 15*x**3 + 15*x**2 + 6*x + 1  # y3(x) = 15x^3 + 15x^2 + 6x + 1
        if self.degree >= 4:
            bessel[:, :, 4] = 105*x**4 + 105*x**3 + 45*x**2 + 10*x + 1  # y4(x) = 105x^4 + 105x^3 + 45x^2 + 10x + 1
        if self.degree >= 5:
            bessel[:, :, 5] = 945*x**5 + 945*x**4 + 420*x**3 + 105*x**2 + 15*x + 1  # y5(x)

        # 使用einsum进行贝塞尔插值
        y = torch.einsum('bid,iod->bo', bessel, self.bessel_coeffs)
        y = y.view(-1, self.output_dim)
        return y

class BesselKAN(nn.Module):
    """
    表示基于贝塞尔多项式的Kolmogorov-Arnold网络(KAN)。
    """
    
    def __init__(self, network, degree):
        """
        初始化BesselKAN网络。

        Args:
            network (list[int]): 定义每层节点数的列表。
            degree (int): 每层使用的贝塞尔多项式的阶数。
        """
        super(BesselKAN, self).__init__()
        self.network = network
        self.layers = nn.ModuleList()

        # 根据指定的网络架构定义层
        for i in range(len(network) - 1):
            self.layers.append(BesselKANLayer(network[i], network[i + 1], degree))

    def forward(self, x):
        """
        网络的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, input_dim)。

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, output_dim)。
        """
        x = x.view(-1, self.network[0])  # 将输入重塑以匹配第一层

        # 顺序通过所有层
        for layer in self.layers:
            x = layer(x)
        return x

class BesselPINN(nn.Module):
    """
    表示使用贝塞尔多项式基的物理信息神经网络(PINN)。
    """

    def __init__(self, network, degree):
        """
        初始化BesselPINN模型。

        Args:
            network (list[int]): 定义每层节点数的列表。
            degree (int): 贝塞尔多项式的阶数。
        """
        super(BesselPINN, self).__init__()
        self.model = BesselKAN(network, degree)  # 定义核心BesselKAN网络
        self.network = network
        self.layers = nn.ModuleList()

    def forward(self, x, x_min=0, x_max=1):
        """
        BesselPINN模型的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为(batch_size, input_dim)。
            x_min (float, optional): 归一化的最小值 (默认: 0)。
            x_max (float, optional): 归一化的最大值 (默认: 1)。

        Returns:
            torch.Tensor: 输出张量，形状为(batch_size, output_dim)。
        """
        return self.model(x)