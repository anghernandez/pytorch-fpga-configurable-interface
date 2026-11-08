import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EVALUATION_DIR / "results"
PROJECT_DIR = EVALUATION_DIR.parent

# --------------------------------------------------
# 1. Arquitectura de referencia
# --------------------------------------------------

class LeNet5(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():

    # --------------------------------------------------
    # 2. Configuración
    # --------------------------------------------------

    device = torch.device("cpu")

    output_dir = RESULTS_DIR / "pytorch_lenet5"
    output_dir.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------
    # 3. Dataset MNIST
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
    # 4. Modelo y pesos entrenados
    # --------------------------------------------------

    model = LeNet5(num_classes=10).to(device)

    state_dict = torch.load(
        "weights/lenet5_reference.pth",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("PyTorch weights loaded successfully.")

    # --------------------------------------------------
    # 5. Variables de evaluación
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
    # 6. Evaluación de MNIST
    # --------------------------------------------------

    with torch.inference_mode():

        for index in range(len(test_dataset)):

            image, real_label = test_dataset[index]

            input_tensor = image.unsqueeze(0).to(device)

            output = model(input_tensor)

            logits.append(output[0].detach().cpu().numpy().copy())

            predicted_label = output.argmax(
                dim=1
            ).item()

            confusion_matrix[
                real_label,
                predicted_label
            ] += 1

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

    print("Logits Pytorch guardados correctamente.")

    # --------------------------------------------------
    # 7. Resultados generales
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
    # 8. Accuracy por clase
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
    # 9. Guardar predicciones
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
    # 10. Guardar matriz de confusión
    # --------------------------------------------------

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        confusion_matrix,
        delimiter=",",
        fmt="%d"
    )

    # --------------------------------------------------
    # 11. Guardar accuracy por clase
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
    # 12. Guardar resumen
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
            "PyTorchLeNet5",
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
