import csv
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from models.CppLeNet5 import CppLeNet5

from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EVALUATION_DIR / "results"
PROJECT_DIR = EVALUATION_DIR.parent


def main() -> None:

    # --------------------------------------------------
    # 1. Configuración
    # --------------------------------------------------

    output_dir = RESULTS_DIR / "cpp_lenet5"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Cargar MNIST
    # --------------------------------------------------

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    # --------------------------------------------------
    # 3. Cargar modelo C++
    # --------------------------------------------------

    model = CppLeNet5()

    model.load_pytorch_weights(
        "weights/lenet5_reference.pth"
    )

    model.eval()

    # --------------------------------------------------
    # 4. Variables de evaluación
    # --------------------------------------------------

    num_classes = 10

    confusion_matrix = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    correct_predictions = 0
    total_predictions = 0

    predictions = []
    logits = []
    # --------------------------------------------------
    # 5. Evaluar las 10 000 imágenes
    # --------------------------------------------------

    with torch.inference_mode():

        for index in range(len(test_dataset)):

            image, real_label = test_dataset[index]

            # Batch size = 1
            input_tensor = image.unsqueeze(0)

            # Inferencia con kernels C++
            output = model(input_tensor)

            # Guardar los 10 logits de la imagen
            logits.append(
                output[0].detach().cpu().numpy().copy()
            )

            # Clase predicha
            predicted_label = output.argmax(
                dim=1
            ).item()

            # Actualizar matriz de confusión
            confusion_matrix[
                real_label,
                predicted_label
            ] += 1

            # Actualizar accuracy
            if predicted_label == real_label:
                correct_predictions += 1

            total_predictions += 1

            predictions.append([
                index,
                real_label,
                predicted_label,
                int(predicted_label == real_label)
            ])

            if (index + 1) % 1000 == 0:
                print(
                    f"Procesadas: {index + 1}"
                    f"/{len(test_dataset)}"
                )


    # --------------------------------------------------
    # Guardar logits de las 10 000 imágenes
    # --------------------------------------------------

    logits_array = np.asarray(
        logits,
        dtype=np.float32
    )

    assert logits_array.shape == (10000, 10), (
        f"Dimensiones inesperadas: {logits_array.shape}"
    )

    np.save(
        output_dir / "logits.npy",
        logits_array
    )

    print("Logits C++ guardados correctamente.")


    # --------------------------------------------------
    # 6. Accuracy global
    # --------------------------------------------------

    accuracy = (
        correct_predictions
        / total_predictions
        * 100
    )

    incorrect_predictions = (
        total_predictions - correct_predictions
    )

    print("\n========== RESULTADOS ==========")

    print("Total de imágenes:", total_predictions)

    print(
        "Predicciones correctas:",
        correct_predictions
    )

    print(
        "Predicciones incorrectas:",
        incorrect_predictions
    )

    print(f"Accuracy global: {accuracy:.4f}%")

    print("\nMatriz de confusión:")
    print(confusion_matrix)

    # --------------------------------------------------
    # 7. Accuracy por clase
    # --------------------------------------------------

    print("\nAccuracy por clase:")

    class_results = []

    for class_index in range(num_classes):

        total_class = confusion_matrix[
            class_index
        ].sum()

        correct_class = confusion_matrix[
            class_index,
            class_index
        ]

        class_accuracy = (
            correct_class / total_class * 100
            if total_class > 0
            else 0.0
        )

        class_results.append([
            class_index,
            int(total_class),
            int(correct_class),
            float(class_accuracy)
        ])

        print(
            f"Clase {class_index}: "
            f"{class_accuracy:.4f}%"
        )

    # --------------------------------------------------
    # 8. Guardar predicciones
    # --------------------------------------------------

    with open(
        output_dir / "predictions.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_index",
            "real_label",
            "predicted_label",
            "correct"
        ])

        writer.writerows(predictions)

    # --------------------------------------------------
    # 9. Guardar matriz de confusión
    # --------------------------------------------------

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        confusion_matrix,
        delimiter=",",
        fmt="%d"
    )

    # --------------------------------------------------
    # 10. Guardar accuracy por clase
    # --------------------------------------------------

    with open(
        output_dir / "accuracy_per_class.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "class",
            "total_images",
            "correct_predictions",
            "accuracy_percent"
        ])

        writer.writerows(class_results)

    # --------------------------------------------------
    # 11. Guardar resumen
    # --------------------------------------------------

    with open(
        output_dir / "summary.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "implementation",
            "dataset",
            "batch_size",
            "total_images",
            "correct_predictions",
            "incorrect_predictions",
            "accuracy_percent"
        ])

        writer.writerow([
            "CppLeNet5",
            "MNIST_test",
            1,
            total_predictions,
            correct_predictions,
            incorrect_predictions,
            accuracy
        ])

    print("\nResultados guardados en:", output_dir)


if __name__ == "__main__":
    main()