# Librerías principales
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split


# Hiperparámetros
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10


# Por ahora usamos CPU
device = torch.device("cpu")

print(f"Using device: {device}")


# Transformaciones para MNIST
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])


# Dataset completo de entrenamiento
train_val_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=mnist_transform,
    download=True
)


# Dataset de test
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=mnist_transform,
    download=True
)


# Separar entrenamiento y validación
train_size = 50000
val_size = 10000

train_dataset, val_dataset = random_split(
    train_val_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


# DataLoader de entrenamiento
train_loader = DataLoader( # Aprende pesos
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# DataLoader de validación
val_loader = DataLoader( # Revisa como va el modelo durante el desarrollo
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# DataLoader de test
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# Definición de LeNet-5 # entra [batch,1,28,28]
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Extracción de características
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2
            ), # [batch ,6,28,28]
            nn.Tanh(),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ), #reduce a la mitad [batch ,6,16,16]

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0   
            ), #[batch,16,10,10]
            nn.Tanh(),

            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            ) #[batch,16,5,5]
        )

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=16 * 5 * 5,
                out_features=120
            ),
            nn.Tanh(),

            nn.Linear(
                in_features=120,
                out_features=84
            ),
            nn.Tanh(),

            nn.Linear(
                in_features=84,
                out_features=num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

model = LeNet5(num_classes=num_classes).to(device)

print(model)

images, labels = next(iter(train_loader))

images = images.to(device)

outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)


# Función de pérdida
criterion = nn.CrossEntropyLoss()

# Optimizador
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

for epoch in range(num_epochs):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(outputs, labels)

        # Limpiar gradientes anteriores
        optimizer.zero_grad()

        # Backward
        loss.backward()

        # Actualizar pesos
        optimizer.step()

        # Acumular loss
        train_loss += loss.item()

        # Calcular accuracy de training
        predictions = outputs.argmax(dim=1)

        train_total += labels.size(0)
        train_correct += (
            predictions == labels
        ).sum().item()

    average_train_loss = train_loss / len(train_loader)

    train_accuracy = (
        100 * train_correct / train_total
    )


        # Validation
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

    average_val_loss = (
        val_loss / len(val_loader)
    )

    val_accuracy = (
        100 * val_correct / val_total
    )

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"| Train Loss: {average_train_loss:.4f} "
        f"| Train Accuracy: {train_accuracy:.2f}% "
        f"| Val Loss: {average_val_loss:.4f} "
        f"| Val Accuracy: {val_accuracy:.2f}%"
    )

# Evaluación final con el conjunto de test
model.eval()

test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        test_total += labels.size(0)

        test_correct += (
            predictions == labels
        ).sum().item()


average_test_loss = (
    test_loss / len(test_loader)
)

test_accuracy = (
    100 * test_correct / test_total
)


print(
    f"Test Loss: {average_test_loss:.4f} "
    f"| Test Accuracy: {test_accuracy:.2f}%"
)


torch.save(
    model.state_dict(),
    "lenet5_pytorch.pth"
)

print("Model weights saved successfully.")


state_dict = model.state_dict()

for name, tensor in state_dict.items():
    print(f"{name:25} {tuple(tensor.shape)}")