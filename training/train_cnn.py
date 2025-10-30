import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
import os

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
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# === Preparar transformaciones ===
def _to_list(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        return [float(i) for i in x]
    raise ValueError(f"Normalize values must be number or list/tuple, got {type(x)}")

def _ensure_channels(lst, channels, name):
    if lst is None:
        raise ValueError(f"{name} is required for Normalize")
    if len(lst) == channels:
        return lst
    if len(lst) == 1:
        return [lst[0]] * channels
    raise ValueError(f"{name} length ({len(lst)}) does not match input_channels ({channels})")

transform_list = []
for t in config["dataset"]["transform"]:
    if isinstance(t, str) and t == "ToTensor":
        transform_list.append(transforms.ToTensor())
        # Si la imagen es mono-canal y el modelo espera más canales, repetir:
        transform_list.append(
            transforms.Lambda(
                lambda x, ch=channels: x.repeat(ch, 1, 1) if x.ndim == 3 and x.shape[0] == 1 and ch > 1 else x
            )
        )

    elif isinstance(t, dict) and "Normalize" in t:
        val = t["Normalize"]
        # Acepta dos formatos comunes:
        # 1) {"Normalize": {"mean": [...], "std": [...]} }
        # 2) {"Normalize": [mean_list, std_list]}
        if isinstance(val, dict):
            mean_raw = _to_list(val.get("mean"))
            std_raw = _to_list(val.get("std"))
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            mean_raw = _to_list(val[0])
            std_raw = _to_list(val[1])
        else:
            raise ValueError("Normalize must be dict with 'mean' and 'std' or a 2-element list [mean, std]")

        mean = _ensure_channels(mean_raw, channels, "mean")
        std = _ensure_channels(std_raw, channels, "std")

        transform_list.append(transforms.Normalize(mean=mean, std=std))

transform = transforms.Compose(transform_list)

# === Cargar datos simulados ===
channels = config["model"]["input_channels"]
image_size = (channels, 32, 32)

dataset = datasets.FakeData(
    size=1000,
    image_size=image_size,
    num_classes=config["model"]["num_classes"],
    transform=transform
)

loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
# === Validación de forma del batch ===
for X_batch, y_batch in loader:
    print("✅ Shape del batch:", X_batch.shape)  # Esperado: [batch_size, 3, 32, 32]
    break
    
# === Inicializar modelo, optimizador y función de pérdida ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(config["model"]["input_channels"], config["model"]["num_classes"]).to(device)

optimizer = getattr(optim, config["training"]["optimizer"])(model.parameters(), lr=config["training"]["learning_rate"])
criterion = getattr(nn, config["training"]["loss_function"])()

# === Entrenamiento ===
for epoch in range(config["training"]["epochs"]):
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
    print(f"Epoch {epoch+1}/{config['training']['epochs']} - Loss: {total_loss:.4f}")

# === Guardar modelo entrenado ===
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/modelo_entrenado.pt")
print("✅ Modelo guardado en models/modelo_entrenado.pt")

