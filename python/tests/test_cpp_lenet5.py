import argparse
import math

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models import CppLeNet5


RMSE_LIMIT = 1e-3
WEIGHTS_PATH = "weights/lenet5_reference.pth"


class LeNet5PyTorch(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Tanh(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.Tanh(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        output_tensor = self.features(
            input_tensor
        )

        return self.classifier(
            output_tensor
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara LeNet-5 PyTorch "
            "contra LeNet-5 con kernels C++"
        )
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Cantidad de imágenes MNIST a comparar"
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1307,),
            std=(0.3081,)
        )
    ])

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    num_images = arguments.num_images

    if (
        num_images <= 0
        or num_images > len(test_dataset)
    ):
        raise ValueError(
            "num_images debe estar entre 1 y "
            f"{len(test_dataset)}"
        )

    state_dict = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )

    #Modelo de referencia PyTorch
    pytorch_model = LeNet5PyTorch()
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    cpp_model = CppLeNet5()
    cpp_model.load_pytorch_weights(
        WEIGHTS_PATH
    )

    cpp_model.eval()

    print("Test dataset size:", len(test_dataset))
    print("Images to compare:", num_images)

    first_image, first_label = test_dataset[0]

    print("First image shape:", first_image.shape)
    print("First image label:", first_label)

    same_predictions = 0

    pytorch_correct = 0
    cpp_correct = 0

    rmse_sum = 0.0
    max_rmse = 0.0
    max_absolute_error = 0.0

    if num_images <= 100:
        progress_interval = 1
    else:
        progress_interval = max(
            1,
            num_images // 100
        )

    for index in range(num_images):
        image, label = test_dataset[index]

        image = image.unsqueeze(0)

        with torch.inference_mode():
            pytorch_output = pytorch_model(image)
            cpp_output = cpp_model(image)

        pytorch_logits = (
            pytorch_output[0].tolist()
        )

        cpp_logits = cpp_output[0].tolist()

        pytorch_prediction = (
            pytorch_output.argmax(dim=1).item()
        )

        cpp_prediction = (
            cpp_output.argmax(dim=1).item()
        )

        if pytorch_prediction == cpp_prediction:
            same_predictions += 1

        if pytorch_prediction == label:
            pytorch_correct += 1

        if cpp_prediction == label:
            cpp_correct += 1

        squared_error_sum = 0.0
        image_max_error = 0.0

        for pytorch_value, cpp_value in zip(
            pytorch_logits,
            cpp_logits
        ):
            error = abs(
                pytorch_value - cpp_value
            )

            squared_error_sum += error ** 2

            if error > image_max_error:
                image_max_error = error

        image_rmse = math.sqrt(
            squared_error_sum
            / len(pytorch_logits)
        )

        rmse_sum += image_rmse

        if image_rmse > max_rmse:
            max_rmse = image_rmse

        if image_max_error > max_absolute_error:
            max_absolute_error = image_max_error

        should_print = (
            (index + 1) % progress_interval == 0
            or index == 0
            or index + 1 == num_images
        )

        if should_print:
            print(
                f"Image {index + 1:5}/{num_images} | "
                f"Label: {label} | "
                f"PyTorch: {pytorch_prediction} | "
                f"C++: {cpp_prediction} | "
                f"RMSE: {image_rmse:.3e}"
            )

    average_rmse = rmse_sum / num_images

    pytorch_accuracy = (
        100 * pytorch_correct / num_images
    )

    cpp_accuracy = (
        100 * cpp_correct / num_images
    )

    validation_passed = (
        max_rmse <= RMSE_LIMIT
    )

    print("\n" + "=" * 60)
    print("LeNet-5 PyTorch vs C++ - Test Summary")
    print("=" * 60)

    print(f"Images tested:             {num_images}")

    print(
        f"Same predictions:          "
        f"{same_predictions}/{num_images}"
    )

    print(
        f"PyTorch accuracy:          "
        f"{pytorch_accuracy:.2f}%"
    )

    print(
        f"C++ accuracy:              "
        f"{cpp_accuracy:.2f}%"
    )

    print(
        f"Average RMSE:              "
        f"{average_rmse:.8e}"
    )

    print(
        f"Maximum RMSE:              "
        f"{max_rmse:.8e}"
    )

    print(
        f"Maximum absolute error:    "
        f"{max_absolute_error:.8e}"
    )

    print(
        f"RMSE limit:                "
        f"{RMSE_LIMIT:.8e}"
    )

    print(
        "Validation:                "
        + (
            "PASSED"
            if validation_passed
            else "FAILED"
        )
    )

    if not validation_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
