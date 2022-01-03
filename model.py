from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


class QuantWeightLeNet(Module):
    def __init__(self):
        super(QuantWeightLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(3, 6, 3, bias=False, weight_bit_width=3, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(6, 12, 3, bias_quant=BiasQuant, weight_bit_width=3, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv3 = qnn.QuantConv2d(12, 18, 3, bias_quant=BiasQuant, weight_bit_width=3, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv4 = qnn.QuantConv2d(18, 18, 3, bias_quant=BiasQuant, weight_bit_width=3, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv5 = qnn.QuantConv2d(18, 18, 3, bias_quant=BiasQuant, weight_bit_width=3, return_quant_tensor=True)
        self.relu5 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc1   = qnn.QuantLinear(162, 120, bias=True, bias_quant=BiasQuant, weight_bit_width=3, return_quant_tensor=True)
        self.relu6 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, bias_quant=BiasQuant, weight_bit_width=3, return_quant_tensor=True)
        self.relu7 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc3   = qnn.QuantLinear(84, 9, bias=False, weight_bit_width=3)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.relu3(self.conv3(out))
        out = F.max_pool2d(out, 2)
        out = self.relu4(self.conv4(out))
        out = F.max_pool2d(out, 2)
        out = self.relu5(self.conv5(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu6(self.fc1(out))
        out = self.relu7(self.fc2(out))
        out = self.fc3(out)
        return out

