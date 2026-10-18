"""Prueba integral pendiente de ejecutar en la Kria KV260.

Valida un único DAG que utiliza los siete nuevos wrappers y los dos anteriores.
"""
import torch
from torch import nn
import torch.nn.functional as F
from fpga_graph import FpgaGraph
from Conv2d import FpgaConv2D
from AvgPool2d import FpgaAvgPool2D
from PointwiseConv2d import FpgaPointwiseConv2d
from DepthwiseConv2d import FpgaDepthwiseConv2d
from BatchNorm2d import FpgaBatchNorm2d
from LayerAdd import FpgaLayerAdd
from ReLU6 import FpgaReLU6
from Tanh import FpgaTanh


class FPGAChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = FpgaConv2D(2, 2, kernel_size=3, padding=1)
        self.bn = FpgaBatchNorm2d(2)
        self.relu = FpgaReLU6()
        self.pw = FpgaPointwiseConv2d(2, 2)
        self.dw = FpgaDepthwiseConv2d(2, kernel_size=3, padding=1)
        self.add = FpgaLayerAdd(alpha=0.75)
        self.pool = FpgaAvgPool2D(kernel_size=2, stride=2)
        self.tanh = FpgaTanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        skip = self.pw(x)
        x = self.dw(skip)
        x = self.add(x, skip)
        x = self.pool(x)
        x = self.tanh(x)
        return x


def main():
    torch.manual_seed(7)
    model = FPGAChain().eval()
    inputs = [torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)]
    graph = FpgaGraph().build(model, inputs[0])
    for index, x in enumerate(inputs, 1):
        with torch.inference_mode():
            y = graph.run(x)
        z = F.conv2d(x, model.conv.weight, model.conv.bias, padding=1)
        z = F.batch_norm(z, model.bn.running_mean, model.bn.running_var,
                         model.bn.weight, model.bn.bias, False, 0., model.bn.eps)
        z = F.relu6(z)
        skip = F.conv2d(z, model.pw.weight, model.pw.bias)
        z = F.conv2d(skip, model.dw.weight, model.dw.bias, padding=1, groups=2)
        z = z + 0.75*skip
        z = F.avg_pool2d(z, 2)
        z = torch.tanh(z)
        rmse = (y-z).square().mean().sqrt().item()
        ok = rmse < 1e-4
        print(f'[DAG wrappers: inferencia {index}] RMSE: {rmse:.6g} | {"PASS" if ok else "FAIL"}')
        assert ok
    print('[DAG wrappers] Todas las pruebas: PASS')


if __name__ == '__main__':
    main()