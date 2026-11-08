import csv
import numpy as np

from pathlib import Path

# Directorio evaluation/
EVALUATION_DIR = Path(__file__).resolve().parent.parent

# Directorio de resultados
RESULTS_DIR = EVALUATION_DIR / "results"


def load_predictions(path):
    with open(path, newline="") as file:
        return {
            int(row["image_index"]): (
                int(row["real_label"]),
                int(row["predicted_label"])
            )
            for row in csv.DictReader(file)
        }


def print_table(headers, rows, widths):
    separator = "+" + "+".join(
        "-" * (width + 2) for width in widths
    ) + "+"

    print(separator)

    print(
        "| " +
        " | ".join(
            str(value).ljust(width)
            for value, width in zip(headers, widths)
        ) +
        " |"
    )

    print(separator)

    for row in rows:
        print(
            "| " +
            " | ".join(
                str(value).ljust(width)
                for value, width in zip(row, widths)
            ) +
            " |"
        )

    print(separator)


def main():

    cpp = load_predictions(
        RESULTS_DIR / "cpp_lenet5/predictions.csv"

    )

    pytorch = load_predictions(
        RESULTS_DIR / "pytorch_lenet5/predictions.csv"

    )

    assert cpp.keys() == pytorch.keys(), (
        "Los conjuntos de imágenes no coinciden"
    )

    different = []

    cpp_correct = 0
    pytorch_correct = 0

    for index in sorted(cpp):

        real_cpp, predicted_cpp = cpp[index]
        real_pytorch, predicted_pytorch = pytorch[index]

        assert real_cpp == real_pytorch, (
            f"Etiqueta real diferente en imagen {index}"
        )

        if predicted_cpp == real_cpp:
            cpp_correct += 1

        if predicted_pytorch == real_pytorch:
            pytorch_correct += 1

        if predicted_cpp != predicted_pytorch:
            different.append(index)

    total = len(cpp)

    if total == 0:
        raise ValueError("No hay predicciones para comparar")

    matches = total - len(different)

    cpp_accuracy = 100 * cpp_correct / total
    pytorch_accuracy = 100 * pytorch_correct / total

    coincidence = 100 * matches / total

    # --------------------------------------------------
    # Validacion numerica: PyTorch vs C++
    # --------------------------------------------------

    pytorch_logits = np.load(
        RESULTS_DIR / "pytorch_lenet5/logits.npy"

    )

    cpp_logits = np.load(
        RESULTS_DIR / "cpp_lenet5/logits.npy"

    )

    # Verificar dimensiones
    assert pytorch_logits.shape == cpp_logits.shape, (
        "Las dimensiones de los logits no coinciden"
    )

    assert pytorch_logits.shape == (total, 10), (
        "Dimensiones inesperadas de los logits"
    )

    # Verificar valores numericos
    assert np.isfinite(pytorch_logits).all(), (
        "PyTorch contiene NaN o Inf"
    )

    assert np.isfinite(cpp_logits).all(), (
        "C++ contiene NaN o Inf"
    )

    # Calcular diferencias en float64
    difference = (
        cpp_logits.astype(np.float64)
        - pytorch_logits.astype(np.float64)
    )

    # Root Mean Squared Error
    rmse = np.sqrt(
        np.mean(difference ** 2)
    )

    # Mean Absolute Error
    mae = np.mean(
        np.abs(difference)
    )

    # Error absoluto maximo
    max_error = np.max(
        np.abs(difference)
    )

    # Umbral de validacion numerica
    tolerance = 1e-3

    rmse_status = (
        "PASS" if rmse <= tolerance else "FAIL"
    )

    # --------------------------------------------------
    # Reporte
    # --------------------------------------------------

    print("\n" + "*" * 60)
    print("          REPORTE DE COMPARACION - LENET-5")
    print("                 AMD KRIA KV260")
    print("*" * 60)

    print("\n[1] CONFIGURACION EXPERIMENTAL\n")

    print_table(
        ["Parametro", "Valor"],
        [
            ["Dataset", "MNIST Test"],
            ["Imagenes evaluadas", total],
            ["Batch size", 1],
            ["Plataforma", "AMD Kria KV260"],
            ["Procesador", "ARM Cortex-A53"],
            ["Implementaciones", "PyTorch / C++"]
        ],
        [24, 38]
    )

    print("\n[2] RESULTADOS DE CLASIFICACION\n")

    print_table(
        ["Metrica", "PyTorch", "C++"],
        [
            ["Imagenes evaluadas", total, total],
            [
                "Predicciones correctas",
                pytorch_correct,
                cpp_correct
            ],
            [
                "Predicciones incorrectas",
                total - pytorch_correct,
                total - cpp_correct
            ],
            [
                "Accuracy (%)",
                f"{pytorch_accuracy:.4f}",
                f"{cpp_accuracy:.4f}"
            ]
        ],
        [27, 16, 16]
    )

    print("\n[3] COMPARACION ENTRE IMPLEMENTACIONES\n")

    print_table(
        ["Metrica", "Resultado"],
        [
            ["Predicciones iguales", matches],
            ["Predicciones diferentes", len(different)],
            ["Coincidencia (%)", f"{coincidence:.4f}"],
            [
                "Diferencia accuracy (pp)",
                f"{abs(pytorch_accuracy - cpp_accuracy):.4f}"
            ]
        ],
        [32, 25]
    )

    print("\n[4] VERIFICACION FUNCIONAL\n")

    if len(different) == 0:
        status = "PASS"
        description = "Todas las predicciones coinciden"
    else:
        status = "DIFFERENCES"
        description = "Existen predicciones diferentes"

    print_table(
        ["Verificacion", "Estado"],
        [
            ["Comparacion de clases", status],
            ["Resultado", description]
        ],
        [28, 40]
    )

    if different:
        print("\n[5] IMAGENES CON DIFERENCIAS\n")

        print_table(
            ["Indice", "Etiqueta real", "PyTorch", "C++"],
            [
                [
                    index,
                    pytorch[index][0],
                    pytorch[index][1],
                    cpp[index][1]
                ]
                for index in different[:20]
            ],
            [12, 16, 12, 12]
        )

        if len(different) > 20:
            print(
                f"\nSe muestran 20 de "
                f"{len(different)} diferencias."
            )

    print("\n[6] VALIDACION NUMERICA - LOGITS\n")

    print_table(
        ["Metrica", "Resultado"],
        [
            ["Valores comparados", difference.size],
            ["RMSE", f"{rmse:.10e}"],
            ["MAE", f"{mae:.10e}"],
            ["Error absoluto maximo", f"{max_error:.10e}"],
            ["Umbral RMSE", f"{tolerance:.10e}"],
            ["Validacion RMSE", rmse_status]
        ],
        [32, 25]
    )


if __name__ == "__main__":
    main()
