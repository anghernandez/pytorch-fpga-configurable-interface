"""Prueba del reshape simbólico; no necesita ejecutar kernels FPGA."""
import torch
import fpga_dag
from fpga_graph import FpgaGraph, FpgaTensor


def main():
    graph = FpgaGraph()
    x = FpgaTensor(graph, 'pool2_output', (1, 16, 5, 5))
    y = x.reshape(x.shape[0], 16 * 5 * 5)
    assert y.shape == (1, 400)
    assert y.tensor_id == x.tensor_id
    assert y.graph is graph
    z = x.reshape(1, -1)
    assert z.shape == (1, 400)

    for invalid in ((1, 399), (0, 400), (-1, -1)):
        try:
            x.reshape(*invalid)
        except ValueError:
            continue
        raise AssertionError(f'Se aceptó forma inválida: {invalid}')

    print('[FpgaTensor.reshape: metadatos y validaciones] PASS')


if __name__ == '__main__':
    main()