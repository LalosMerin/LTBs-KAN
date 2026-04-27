#  Training=====================
#  =====================

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import time
import matplotlib.pyplot as plt


# ===================== TRANSFORMACIÓN Y DATASETS =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# ===================== PÉRDIDA, OPTIMIZADORES, SCHEDULERS =====================
criterion = nn.CrossEntropyLoss()


optimizers = [optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) for model in models]
schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5) for optimizer in optimizers]
#schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25) for optimizer in optimizers]
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# ===================== MÉTRICAS DESDE MATRIZ DE CONFUSIÓN =====================
def _metrics_from_confmat(confmat: torch.Tensor, eps: float = 1e-12):
    """
    confmat: tensor [C, C] donde filas = etiqueta verdadera, columnas = predicción
    Devuelve (precision_macro, recall_macro, f1_macro)
    """
    tp = torch.diag(confmat)                        # [C]
    fp = confmat.sum(dim=0) - tp                    # [C]
    fn = confmat.sum(dim=1) - tp                    # [C]

    precision_c = tp / (tp + fp + eps)
    recall_c    = tp / (tp + fn + eps)
    f1_c        = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

    precision_macro = precision_c.mean().item()
    recall_macro    = recall_c.mean().item()
    f1_macro        = f1_c.mean().item()
    return precision_macro, recall_macro, f1_macro

# ===================== ENTRENAMIENTO =====================
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # MNIST 28x28 -> 784

        if isinstance(optimizer, optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                return loss
            loss_tensor = optimizer.step(closure)
            loss_value = loss_tensor.item() if torch.is_tensor(loss_tensor) else float(loss_tensor)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()

        total_loss += loss_value

    return total_loss / len(train_loader)

# ===================== VALIDACIÓN (CON MÉTRICAS) =====================
def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    num_samples = 0
    num_classes = None
    confmat = None  # [C, C]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            num_samples += target.size(0)

            # Inicializar confmat una vez conozcamos C
            if num_classes is None:
                num_classes = output.size(1)
                confmat = torch.zeros(num_classes, num_classes, device=target.device, dtype=torch.long)

            # ---- Vectorizado para acumular matriz de confusión ----
            idx = target.to(torch.long) * num_classes + pred.to(torch.long)
            binc = torch.bincount(idx, minlength=num_classes * num_classes)
            confmat += binc.view(num_classes, num_classes)

    avg_loss = total_loss / len(test_loader)       # Test loss (se mantiene igual)
    accuracy = correct / num_samples               # Exactitud global

    # Métricas macro desde la confusión
    precision_macro, recall_macro, f1_macro = _metrics_from_confmat(confmat.to(torch.float32))

    return avg_loss, accuracy, precision_macro, recall_macro, f1_macro

# ===================== LOOP: ENTRENAR Y VALIDAR VARIOS MODELOS =====================
def train_and_validate(models, model_names, train_loader, test_loader, criterion, optimizers, schedulers, device, epochs):
    train_losses = [[] for _ in models]
    test_losses  = [[] for _ in models]
    best_weights = [None] * len(models)
    best_test_losses = [float('inf')] * len(models)
    model_times = [0.0] * len(models)

    # Nuevas colecciones de métricas
    test_accuracies = [[] for _ in models]
    test_precisions = [[] for _ in models]  # macro
    test_recalls    = [[] for _ in models]  # macro
    test_f1s        = [[] for _ in models]  # macro

    for epoch in range(epochs):
        for i, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
            start_time = time.time()

            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, test_prec, test_rec, test_f1 = validate(model, test_loader, criterion, device)

            model_times[i] += time.time() - start_time
            train_losses[i].append(train_loss)
            test_losses[i].append(test_loss)
            test_accuracies[i].append(test_acc)
            test_precisions[i].append(test_prec)
            test_recalls[i].append(test_rec)
            test_f1s[i].append(test_f1)

            if test_loss < best_test_losses[i]:
                best_test_losses[i] = test_loss
                best_weights[i] = model.state_dict()

            scheduler.step()  # Actualiza learning rate

            print(
                f'{model_names[i]} | Epoch {epoch+1}/{epochs} | '
                f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | '
                f'Acc: {test_acc*100:.2f}% | P_macro: {test_prec:.4f} | '
                f'R_macro: {test_rec:.4f} | F1_macro: {test_f1:.4f} | '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )

    return (train_losses, test_losses, best_weights, model_times,
            test_accuracies, test_precisions, test_recalls, test_f1s)



epochs = 15


Model_Names=['LTBs-KAN'] # Add names of other models

model0 = KAN_NM([28 * 28, 32, 10])

# Add more models here if needed, e.g.:
models = [model0]



for model in models:
    model.to(device)

(train_losses, test_losses, best_weights, model_times,
 test_accuracies, test_precisions, test_recalls, test_f1s) = train_and_validate(
    models, Model_Names, train_loader, test_loader, criterion, optimizers, schedulers, device, epochs)


    plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
})

plt.figure(figsize=(10, 5), facecolor='white')

for i in range(len(models)):
    plt.plot(range(1, epochs + 1), test_losses[i], label=f'{Model_Names[i]}', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(frameon=False)  # leyenda sin fondo gris

# Grid más limpio (no gris pesado)
plt.grid(True, color='black', alpha=0.1)

plt.tight_layout()

plt.savefig("testloss_MNIST.pdf",format="pdf",facecolor='white', bbox_inches='tight',pad_inches=0)


plt.show()
# best_weights es una lista de diccionarios de estado para cada modelo. Guardamos cada uno en un archivo separado.
for i, model in enumerate(models):
    model.load_state_dict(best_weights[i])
    torch.save(model.state_dict(), f'{Model_Names[i]}_best_weights.pth')

# Mostrar tiempos
for i, name in enumerate(Model_Names):
    print(f'{name} total training time: {model_times[i]:.2f} seconds')