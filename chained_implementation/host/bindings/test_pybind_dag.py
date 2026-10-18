"""Valida desde Python dos inferencias sobre un DAG diferido ReLU6 -> Tanh."""
import numpy as np
import fpga_dag


def main():
    # 1. Construir el grafo una sola vez (sin ejecutar kernels).
    fpga_dag.clear_graph()
    fpga_dag.add_input("py_entrada")
    fpga_dag.add_operation("ReLU6", ["py_entrada"], [], "py_relu", {"size": 7})
    fpga_dag.add_operation("Tanh", ["py_relu"], [], "py_salida", {"size": 7})
    fpga_dag.set_outputs(["py_salida"])
    fpga_dag.finalize_graph()

    # 2. Ejecutar el mismo grafo con entradas distintas.
    for step, input_values in enumerate((
        np.array([-3, -1, 0, 1, 3, 6, 8], dtype=np.float32),
        np.array([-5, 0.25, 2, 4, 6, 7, 10], dtype=np.float32),
    ), start=1):
        fpga_dag.upload_input("py_entrada", input_values, "relu6_forward", 0)
        fpga_dag.execute_graph()
        output = fpga_dag.download_output("py_salida", [7])
        reference = np.tanh(np.clip(input_values, 0, 6)).astype(np.float32)
        rmse = float(np.sqrt(np.mean((output - reference) ** 2)))
        passed = bool(np.allclose(output, reference, atol=1e-5, rtol=0))
        print(f"[Python DAG: inferencia {step}] RMSE: {rmse:.8g} | {'PASS' if passed else 'FAIL'}")
        if not passed:
            raise AssertionError(f"Inferencia {step} incorrecta")
    print("[Python DAG] Todas las pruebas: PASS")


if __name__ == "__main__":
    main()