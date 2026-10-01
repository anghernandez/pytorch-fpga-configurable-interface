import torch
from torchvision import datasets, transforms

from models import CppLeNet5


def main() -> None:
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

    image, real_label = test_dataset[0]
    input_tensor = image.unsqueeze(0)

    model = CppLeNet5()

    model.load_pytorch_weights(
        "weights/lenet5_reference.pth"
    )

    model.eval()

    with torch.inference_mode():
        output = model(input_tensor)

    prediction = output.argmax(
        dim=1
    ).item()

    print("Forma de entrada:", input_tensor.shape)
    print("Forma de salida:", output.shape)
    print("Etiqueta real:", real_label)
    print("Predicción C++:", prediction)
    print("Logits:", output[0])


if __name__ == "__main__":
    main()
