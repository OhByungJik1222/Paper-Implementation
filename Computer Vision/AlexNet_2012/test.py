import torch
import torch.nn as nn
import yaml

from model import AlexNet
from dataset import get_dataloader

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_dataloader(
    cfg["train_dir"], cfg["val_dir"], cfg["test_dir"], cfg["batch_size"]
)

model = AlexNet(cfg["num_classes"]).to(device)
model.load_state_dict(torch.load("best_alexnet.pth"))

criterion = nn.CrossEntropyLoss()

model.eval()

test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for img, cls in test_loader:
        img, cls = img.to(device), cls.to(device)

        output = model(img)
        loss = criterion(output, cls)

        test_loss += loss.item() * img.size(0)
        _, preds = torch.max(output, 1)
        correct += (preds == cls).sum().item()
        total += cls.size(0)

test_loss /= total
test_acc = correct / total

print(f'Test Loss {test_loss:.4f} - Test Acc {test_acc:.4f}')