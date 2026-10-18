"""Comparación MobileNetV2 ImageNet-1K: torchvision vs FPGA.
Se compara una imagen (1000 logits).
"""
import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from fpga_graph import FpgaGraph
from models.FpgaMobileNetV2 import FpgaMobileNetV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Imagen RGB de prueba")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="Tolerancia máxima absoluta por logit")
    parser.add_argument("--weights", type=Path, default=None,
                        help="state_dict torchvision local; si se omite, torchvision descarga/cachea los pesos oficiales")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Imagen inexistente: {args.image}")

    # Los pesos incluyen la definición del preprocesamiento de ImageNet-1K.
    weights = MobileNet_V2_Weights.DEFAULT
    reference = mobilenet_v2(weights=None).eval()
    if args.weights is None:
        reference = mobilenet_v2(weights=weights).eval()
    else:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        reference.load_state_dict(state, strict=True)

    fpga_model = FpgaMobileNetV2(num_classes=1000).eval()
    fpga_model.load_pytorch_weights(reference)

    with Image.open(args.image) as image:
        input_tensor = weights.transforms()(image.convert("RGB")).unsqueeze(0).contiguous()

    # Build únicamente una vez: carga parámetros persistentes a XRT.
    with torch.inference_mode():
        graph = FpgaGraph().build(fpga_model, torch.zeros_like(input_tensor))
        expected = reference(input_tensor)
        actual = graph.run(input_tensor)

    if expected.shape != (1, 1000) or actual.shape != (1, 1000):
        raise RuntimeError(f"Formas inesperadas: CPU={tuple(expected.shape)}, FPGA={tuple(actual.shape)}")
    difference = (actual.double() - expected.double()).abs()
    rmse = math.sqrt(float(difference.square().mean()))
    max_error = float(difference.max())
    cpu_top5 = expected.topk(5, dim=1).indices[0].tolist()
    fpga_top5 = actual.topk(5, dim=1).indices[0].tolist()
    categories = weights.meta["categories"]

    print(f"Imagen: {args.image}")
    print(f"Entrada: {tuple(input_tensor.shape)} | Salida: {tuple(actual.shape)}")
    print(f"RMSE logits: {rmse:.10e}")
    print(f"Error máximo: {max_error:.10e}")
    print(f"Top-1 PyTorch: {cpu_top5[0]} ({categories[cpu_top5[0]]})")
    print(f"Top-1 FPGA:    {fpga_top5[0]} ({categories[fpga_top5[0]]})")
    print(f"Top-5 PyTorch: {cpu_top5}")
    print(f"Top-5 FPGA:    {fpga_top5}")
    passed = max_error <= args.tolerance
    print(f"[MobileNetV2 vs PyTorch] {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()