import math
import torch
import torchvision

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

from python.models.ManualMobileNetV2 import (
    ManualMobileNetV2,
)


def calculate_rmse(reference, manual):

    if len(reference) != len(manual):
        raise ValueError(
            "Las salidas deben tener el mismo tamaño"
        )

    squared_error_sum = 0.0

    for index in range(len(reference)):

        difference = (
            reference[index]
            - manual[index]
        )

        squared_error_sum += (
            difference * difference
        )

    return math.sqrt(
        squared_error_sum / len(reference)
    )


def calculate_max_error(reference, manual):

    max_error = 0.0

    for index in range(len(reference)):

        error = abs(
            reference[index]
            - manual[index]
        )

        if error > max_error:
            max_error = error

    return max_error


if __name__ == "__main__":

    # ========================================================
    # 1. MobileNetV2 oficial de PyTorch
    # ========================================================

    print()
    print("=" * 70)
    print("CARGANDO MOBILENETV2 OFICIAL")
    print("=" * 70)

    weights = (
        MobileNet_V2_Weights.IMAGENET1K_V1
    )

    pytorch_model = mobilenet_v2(
        weights=weights
    )

    pytorch_model.eval()

    print(
        "Modelo PyTorch cargado."
    )

    # ========================================================
    # 2. MobileNetV2 manual
    # ========================================================

    print()
    print("=" * 70)
    print("CREANDO MOBILENETV2 MANUAL")
    print("=" * 70)

    manual_model = ManualMobileNetV2(
        num_classes=1000
    )

    print(
        "Modelo manual creado."
    )

    # ========================================================
    # 3. Copiar parámetros PyTorch -> Manual
    # ========================================================

    print()
    print("=" * 70)
    print("CARGANDO PESOS PYTORCH EN MODELO MANUAL")
    print("=" * 70)

    manual_model.load_pytorch_weights(
        pytorch_model
    )

    # ========================================================
    # 4. Comprobar estructura
    # ========================================================

    print()
    print("=" * 70)
    print("COMPROBACIÓN DE ESTRUCTURA")
    print("=" * 70)

    print(
        "Número de bloques manuales:",
        len(manual_model.blocks)
    )

    print(
        "Número de features PyTorch:",
        len(pytorch_model.features)
    )

    print(
        "Clases:",
        manual_model.num_classes
    )

    if len(manual_model.blocks) != 17:
        raise RuntimeError(
            "La MobileNetV2 manual no contiene "
            "los 17 bloques esperados"
        )

    print(
        "Estructura básica: PASS"
    )

    # ========================================================
    # 5. Entrada de prueba
    #
    # TODAVÍA no usamos imagen real.
    # Primero queremos comprobar que todo el modelo
    # puede ejecutar correctamente.
    # ========================================================

    print()
    print("=" * 70)
    print("CREANDO ENTRADA")
    print("=" * 70)

    torch.manual_seed(42)

    input_tensor = torch.randn(
        1,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    print(
        "Input shape:",
        tuple(input_tensor.shape)
    )

    # ========================================================
    # 6. PyTorch
    # ========================================================

    print()
    print("=" * 70)
    print("INFERENCIA PYTORCH")
    print("=" * 70)

    with torch.no_grad():

        pytorch_output = pytorch_model(
            input_tensor
        )

    pytorch_logits = (
        pytorch_output[0]
        .detach()
        .cpu()
        .tolist()
    )

    pytorch_prediction = (
        pytorch_output
        .argmax(dim=1)
        .item()
    )

    print(
        "Output shape:",
        tuple(pytorch_output.shape)
    )

    print(
        "Predicción PyTorch:",
        pytorch_prediction
    )

    # ========================================================
    # 7. Manual
    # ========================================================

    print()
    print("=" * 70)
    print("INFERENCIA MANUAL")
    print("=" * 70)

    manual_input = (
        input_tensor
        .detach()
        .cpu()
        .tolist()
    )

    manual_output = manual_model(
        manual_input,
        verbose=True,
    )

    manual_logits = manual_output[0]

    manual_prediction = max(
        range(len(manual_logits)),
        key=lambda index: manual_logits[index],
    )

    print()
    print(
        "Número de logits manuales:",
        len(manual_logits)
    )

    print(
        "Predicción manual:",
        manual_prediction
    )

    # ========================================================
    # 8. Comparación
    # ========================================================

    print()
    print("=" * 70)
    print("COMPARACIÓN PYTORCH VS MANUAL")
    print("=" * 70)

    rmse = calculate_rmse(
        pytorch_logits,
        manual_logits,
    )

    max_error = calculate_max_error(
        pytorch_logits,
        manual_logits,
    )

    print(
        f"RMSE:      {rmse:.10e}"
    )

    print(
        f"Max error: {max_error:.10e}"
    )

    print(
        "Predicción PyTorch:",
        pytorch_prediction
    )

    print(
        "Predicción manual:",
        manual_prediction
    )

    # ========================================================
    # 9. Mostrar algunos logits
    # ========================================================

    print()
    print(
        "Primeros 10 logits:"
    )

    for index in range(10):

        pytorch_value = (
            pytorch_logits[index]
        )

        manual_value = (
            manual_logits[index]
        )

        error = abs(
            pytorch_value
            - manual_value
        )

        print(
            f"Clase {index:4d} | "
            f"PyTorch={pytorch_value: .8f} | "
            f"Manual={manual_value: .8f} | "
            f"Error={error:.8e}"
        )

    # ========================================================
    # 10. Resultado
    # ========================================================

    print()
    print("=" * 70)

    tolerance = 1e-3

    if (
        rmse <= tolerance
        and pytorch_prediction
        == manual_prediction
    ):

        print(
            "RESULTADO FINAL: PASS"
        )

    else:

        print(
            "RESULTADO FINAL: FAIL"
        )
