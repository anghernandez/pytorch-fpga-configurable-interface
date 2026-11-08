import torch
import torch.nn.functional as F
from fpga_graph import FpgaGraph
from BatchNorm2d import FpgaBatchNorm2d

x = torch.tensor(
    [-3.0, -1.0, 0.0, 0.5, 2.0, 5.0, 6.0, 9.0],
    dtype=torch.float32
).reshape(1, 2, 2, 2)

model = FpgaBatchNorm2d(2).eval()

with torch.inference_mode():
    expected = F.batch_norm(
        x, model.running_mean, model.running_var,
        model.weight, model.bias,
        training=False, eps=model.eps
    )
    graph = FpgaGraph().build(model, x)
    actual = graph.run(x)

print("PyTorch:", expected.flatten().tolist())
print("FPGA:   ", actual.flatten().tolist())

passed = actual.shape == expected.shape and torch.allclose(
    actual, expected, atol=1e-4, rtol=0
)
print("BatchNorm sola:", "PASS" if passed else "FAIL")
