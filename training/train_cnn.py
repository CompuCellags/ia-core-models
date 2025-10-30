import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import os
import sys

# === Cargar configuración YAML ===
with open("configs/cnn_default.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Definir modelo CNN simple ===
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === Simular datos de entrenamiento ===
def generate_dummy_data(samples, input_shape, num_classes):
    X = torch.randn(samples, *input_shape)
    y = torch.randint(0, num_classes, (samples,))
    return TensorDataset(X, y)

# === Entrenamiento ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = tuple(config["model"]["input_shape"])
    num_classes = config["model"]["num_classes"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]

    model = SimpleCNN(input_channels=input_shape[0], num_classes=num_classes).to(device)
    dataset = generate_dummy_data(1000, input_shape, num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/modelo_entrenado.pt")
    print("✅ Modelo guardado en models/modelo_entrenado.pt")

if __name__ == "__main__":
    train()
