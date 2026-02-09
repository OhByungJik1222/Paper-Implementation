import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from model import AlexNet
from dataset import get_dataloader

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_dataloader(
    cfg["train_dir"], cfg["val_dir"], cfg["test_dir"], cfg["batch_size"]
)

# Details of Learning (Section 5)
# weight decay: 학습 오류 감소
model = AlexNet(cfg["num_classes"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=cfg["lr"],
    momentum=cfg["momentum"],
    weight_decay=cfg["weight_decay"]
)

lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=cfg["step_size"],
    gamma=cfg["gamma"]
)

# Epoch (Section 5) - Roughly 90
best_acc = 0
for epoch in range(cfg["epochs"]):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for img, cls in train_loader:
        img, cls = img.to(device), cls.to(device)

        output = model(img)
        loss = criterion(output, cls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img.size(0)
        _, preds = torch.max(output, 1)
        correct += (preds == cls).sum().item()
        total += cls.size(0)
    
    train_loss /= total
    train_acc = correct / total

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, cls in val_loader:
            img, cls = img.to(device), cls.to(device)

            output = model(img)
            loss = criterion(output, cls)

            val_loss += loss.item() * img.size(0)
            _, preds = torch.max(output, 1)
            correct += (preds == cls).sum().item()
            total += cls.size(0)

    val_loss /= total
    val_acc = correct / total

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}/90 - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Val Loss {val_loss:.4f} - Val Acc {val_acc:.4f}')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_alexnet.pth")
        print("Best Model Updated")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, "alexnet_checkpoint.pth")

    lr_scheduler.step()