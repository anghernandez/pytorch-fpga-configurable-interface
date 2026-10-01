import math
from pathlib import Path
from PIL import Image

import torch

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

from python.models.ManualMobileNetV2 import (
    ManualMobileNetV2,
)


# ============================================================
# Métricas
# ============================================================

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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("PRUEBA MOBILENETV2 - IMAGENET-1000")
    print("=" * 70)

    # ========================================================
    # 1. Pesos oficiales ImageNet-1K
    # ========================================================

    weights = (
        MobileNet_V2_Weights.IMAGENET1K_V1
    )

    categories = weights.meta[
        "categories"
    ]

    print()
    print(
        "Número de clases:",
        len(categories)
    )

    # ========================================================
    # 2. Modelo oficial PyTorch
    # ========================================================

    print()
    print("=" * 70)
    print("CARGANDO MOBILENETV2 PYTORCH")
    print("=" * 70)

    pytorch_model = mobilenet_v2(
        weights=weights
    )

    pytorch_model.eval()

    print(
        "Modelo PyTorch cargado."
    )

    # ========================================================
    # 3. Modelo manual
    # ========================================================

    print()
    print("=" * 70)
    print("CARGANDO MOBILENETV2 MANUAL")
    print("=" * 70)

    manual_model = ManualMobileNetV2(
        num_classes=1000
    )

    manual_model.load_pytorch_weights(
        pytorch_model
    )

    print(
        "Modelo manual cargado."
    )

    # ========================================================
    # 4. Buscar imágenes
    # ========================================================

    print()
    print("=" * 70)
    print("BUSCANDO IMÁGENES")
    print("=" * 70)

    images_directory = Path(
        "python/tests/images"
    )

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    if not images_directory.exists():

        raise RuntimeError(
            "No existe la carpeta: "
            "python/tests/images"
        )

    image_paths = [
        path
        for path in sorted(
            images_directory.iterdir()
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in valid_extensions
        )
    ]

    if len(image_paths) == 0:

        raise RuntimeError(
            "No se encontraron imágenes "
            "en python/tests/images"
        )

    print(
        "Imágenes encontradas:",
        len(image_paths)
    )

    for image_path in image_paths:

        print(
            " -",
            image_path.name
        )

    # ========================================================
    # 5. Preprocesamiento oficial ImageNet
    # ========================================================

    preprocess = weights.transforms()

    # ========================================================
    # Variables para resultados globales
    # ========================================================

    total_images = 0

    same_top1_count = 0

    same_top5_count = 0

    rmse_sum = 0.0

    global_max_error = 0.0

    # ========================================================
    # 6. Procesar imágenes
    # ========================================================

    for image_number, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print("=" * 70)

        print(
            f"IMAGEN "
            f"{image_number}/"
            f"{len(image_paths)}"
        )

        print("=" * 70)

        print(
            "Archivo:",
            image_path.name
        )

        # ====================================================
        # 6.1 Abrir imagen
        # ====================================================

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        print(
            "Tamaño original:",
            image.size
        )

        # ====================================================
        # 6.2 Preprocesamiento
        # ====================================================

        input_tensor = preprocess(
            image
        ).unsqueeze(0)

        print(
            "Input shape:",
            tuple(input_tensor.shape)
        )

        print(
            "Input dtype:",
            input_tensor.dtype
        )

        # ====================================================
        # 6.3 Inferencia PyTorch
        # ====================================================

        print()
        print(
            "Ejecutando PyTorch..."
        )

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

        pytorch_class = categories[
            pytorch_prediction
        ]

        # ====================================================
        # 6.4 Inferencia manual
        # ====================================================

        print(
            "Ejecutando MobileNetV2 manual..."
        )

        manual_input = (
            input_tensor
            .detach()
            .cpu()
            .tolist()
        )

        manual_output = manual_model(
            manual_input,
            verbose=False,
        )

        manual_logits = manual_output[
            0
        ]

        manual_prediction = max(
            range(
                len(manual_logits)
            ),
            key=lambda index:
                manual_logits[index]
        )

        manual_class = categories[
            manual_prediction
        ]

        # ====================================================
        # 6.5 Comparación numérica
        # ====================================================

        rmse = calculate_rmse(
            pytorch_logits,
            manual_logits,
        )

        max_error = calculate_max_error(
            pytorch_logits,
            manual_logits,
        )

        same_top1 = (
            pytorch_prediction
            == manual_prediction
        )

        # ====================================================
        # 6.6 Top-5 PyTorch
        # ====================================================

        pytorch_top5 = torch.topk(
            pytorch_output[0],
            5
        )

        pytorch_top5_indices = []

        for position in range(5):

            class_index = int(
                pytorch_top5.indices[
                    position
                ]
            )

            pytorch_top5_indices.append(
                class_index
            )

        # ====================================================
        # 6.7 Top-5 manual
        # ====================================================

        manual_top5_indices = sorted(
            range(
                len(manual_logits)
            ),
            key=lambda index:
                manual_logits[index],
            reverse=True,
        )[:5]

        same_top5 = (
            pytorch_top5_indices
            == manual_top5_indices
        )

        # ====================================================
        # 6.8 Mostrar resultados
        # ====================================================

        print()
        print("-" * 70)
        print("RESULTADOS")
        print("-" * 70)

        print()
        print(
            "PyTorch:"
        )

        print(
            f"  Clase:    "
            f"{pytorch_prediction}"
        )

        print(
            f"  Etiqueta: "
            f"{pytorch_class}"
        )

        print()
        print(
            "Manual:"
        )

        print(
            f"  Clase:    "
            f"{manual_prediction}"
        )

        print(
            f"  Etiqueta: "
            f"{manual_class}"
        )

        print()
        print(
            f"RMSE:      "
            f"{rmse:.10e}"
        )

        print(
            f"Max error: "
            f"{max_error:.10e}"
        )

        print(
            "Top-1 coincide:",
            same_top1
        )

        print(
            "Top-5 coincide:",
            same_top5
        )

        # ====================================================
        # 6.9 Mostrar Top-5
        # ====================================================

        print()
        print(
            "TOP-5 PYTORCH"
        )

        for position in range(5):

            class_index = (
                pytorch_top5_indices[
                    position
                ]
            )

            value = float(
                pytorch_top5.values[
                    position
                ]
            )

            print(
                f"  {position + 1}. "
                f"{class_index:4d} | "
                f"{categories[class_index]} | "
                f"logit={value:.6f}"
            )

        print()
        print(
            "TOP-5 MANUAL"
        )

        for position in range(5):

            class_index = (
                manual_top5_indices[
                    position
                ]
            )

            value = (
                manual_logits[
                    class_index
                ]
            )

            print(
                f"  {position + 1}. "
                f"{class_index:4d} | "
                f"{categories[class_index]} | "
                f"logit={value:.6f}"
            )

        # ====================================================
        # 6.10 Acumular resultados
        # ====================================================

        total_images += 1

        rmse_sum += rmse

        if same_top1:

            same_top1_count += 1

        if same_top5:

            same_top5_count += 1

        if max_error > global_max_error:

            global_max_error = (
                max_error
            )

    # ========================================================
    # 7. Resultados globales
    # ========================================================

    average_rmse = (
        rmse_sum
        / total_images
    )

    top1_agreement = (
        100.0
        * same_top1_count
        / total_images
    )

    top5_agreement = (
        100.0
        * same_top5_count
        / total_images
    )

    print()
    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    print(
        "Imágenes procesadas:",
        total_images
    )

    print(
        "Top-1 iguales:",
        f"{same_top1_count}/"
        f"{total_images}"
    )

    print(
        "Coincidencia Top-1:",
        f"{top1_agreement:.2f}%"
    )

    print(
        "Top-5 iguales:",
        f"{same_top5_count}/"
        f"{total_images}"
    )

    print(
        "Coincidencia Top-5:",
        f"{top5_agreement:.2f}%"
    )

    print(
        f"RMSE promedio: "
        f"{average_rmse:.10e}"
    )

    print(
        f"Máximo error global: "
        f"{global_max_error:.10e}"
    )

    # ========================================================
    # 8. PASS / FAIL
    # ========================================================

    tolerance = 1e-3

    if (
        average_rmse <= tolerance
        and same_top1_count
        == total_images
        and same_top5_count
        == total_images
    ):

        print()
        print(
            "RESULTADO FINAL: PASS"
        )

    else:

        print()
        print(
            "RESULTADO FINAL: FAIL"
        )