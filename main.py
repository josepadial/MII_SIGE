import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import mlflow


# Custom Dataset class for loading images and additional data
class ImageDataWithAdditionalDataset(Dataset):
    def __init__(self, image_folder, additional_data_folder, transform=None):
        self.image_data = datasets.ImageFolder(image_folder, transform=transform)
        self.additional_data = self.load_additional_data(additional_data_folder)
        self.transform = transform

    def load_additional_data(self, image_folder):
        additional_data = {}
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    class_name = os.path.basename(root)
                    additional_data[class_name] = file  # Save the filename as additional data
        return additional_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        image, label = self.image_data[index]
        class_name = self.image_data.classes[label]

        additional_info = self.additional_data.get(class_name, "")

        return image, label, additional_info


# Set the paths for image and additional data folders
image_data_path = "data x200"
additional_data_path = "data additional"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the combined dataset
data = ImageDataWithAdditionalDataset(image_data_path, additional_data_path, transform=transform)

# Partition the data into training, validation, and test sets
train_size = int(0.8 * len(data))
val_size = (len(data) - train_size) // 2
test_size = len(data) - train_size - val_size

train_data, val_data, test_data = random_split(data, [train_size, val_size, test_size])

# Define the data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Clasificación multiclase
model1 = models.resnet50(pretrained=True)
model2 = models.densenet121(pretrained=True)
model3 = models.vgg16(pretrained=True)

num_features1 = model1.fc.in_features
num_features2 = model2.classifier.in_features
num_features3 = model3.classifier[6].in_features

model1.fc = nn.Linear(num_features1, len(data.image_data.classes))
model2.classifier = nn.Linear(num_features2, len(data.image_data.classes))
model3.classifier[6] = nn.Linear(num_features3, len(data.image_data.classes))

# Ajuste de hiperparámetros, topología de la red, función de coste y optimizador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fine Tuning
for param in model1.parameters():
    param.requires_grad = False

for param in model1.fc.parameters():
    param.requires_grad = True

for param in model2.parameters():
    param.requires_grad = False

for param in model2.classifier.parameters():
    param.requires_grad = True

for param in model3.parameters():
    param.requires_grad = False

for param in model3.classifier[6].parameters():
    param.requires_grad = True

models = [model1, model2, model3]

criterions = [VariableCostLoss(num_classes=len(data.image_data.classes)) for _ in models]
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

# Aplicar técnicas para mejora del aprendizaje

# 1. Aumento de datos
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data.dataset.image_data.transform = train_transform

# 2. Ajuste de tasa de aprendizaje
schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) for optimizer in optimizers]

# Iniciar un experimento de MLflow
mlflow.start_run()

# Training progress
train_loss_history = []
val_accuracy_history = []
confusion_matrix_val = None

# Entrenamiento del modelo
for epoch in range(10):
    running_loss = [0.0] * len(models)

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        for idx, model in enumerate(models):
            model.train()
            outputs = model(images)
            loss = criterions[idx](outputs, labels)
            loss.backward()
            optimizers[idx].step()

            running_loss[idx] += loss.item()

    for scheduler in schedulers:
        scheduler.step()  # Actualizar la tasa de aprendizaje

    # Calculate validation accuracy and confusion matrix
    for model in models:
        model.eval()

    correct = [0] * len(models)
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = [model(images) for model in models]
            ensemble_outputs = torch.stack(outputs, dim=0).sum(dim=0)
            _, predicted = torch.max(ensemble_outputs.data, 1)
            total += labels.size(0)

            for idx, pred in enumerate(predicted):
                correct[idx] += (pred == labels).sum().item()

            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    accuracy = [100 * correct[i] / total for i in range(len(models))]
    val_accuracy_history.append(accuracy)
    confusion_matrix_val = confusion_matrix(true_labels, predictions, normalize='true')

    train_loss_history.append([loss / len(train_loader) for loss in running_loss])
    print('Epoch [{}/{}], Losses: {}, Val Accuracy: {}'.format(epoch + 1, 10, [loss / len(train_loader) for loss in running_loss], accuracy))

    # Registrar las métricas y parámetros relevantes en MLflow
    for idx in range(len(models)):
        mlflow.log_metric(f"train_loss_model{idx+1}", running_loss[idx] / len(train_loader), step=epoch)
        mlflow.log_metric(f"val_accuracy_model{idx+1}", accuracy[idx], step=epoch)

# Finalizar el experimento de MLflow
mlflow.end_run()

# Plot training progress
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for idx in range(len(models)):
    plt.plot(range(1, len(train_loss_history) + 1), [loss[idx] for loss in train_loss_history], label=f"Model {idx+1}")
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss Progress')
plt.legend()

plt.subplot(1, 2, 2)
for idx in range(len(models)):
    plt.plot(range(1, len(val_accuracy_history) + 1), [accuracy[idx] for accuracy in val_accuracy_history], label=f"Model {idx+1}")
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy Progress')
plt.legend()

plt.tight_layout()

# Guardar la figura en MLflow
fig_path = f"{image_data_path}training_progress.png"
plt.savefig(fig_path)
mlflow.log_artifact(fig_path)

# Mostrar la figura
plt.show()

# Plot confusion matrix
plt.figure(figsize=(10, 8))
class_names = data.image_data.classes
plt.imshow(confusion_matrix_val, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = confusion_matrix_val.max() / 2.0
for i, j in itertools.product(range(confusion_matrix_val.shape[0]), range(confusion_matrix_val.shape[1])):
    plt.text(j, i, "{:.2f}".format(confusion_matrix_val[i, j]),
             horizontalalignment="center",
             color="white" if confusion_matrix_val[i, j] > thresh else "black")

plt.tight_layout()

# Guardar la figura en MLflow
fig_path = f"{image_data_path}confusion_matrix.png"
plt.savefig(fig_path)
mlflow.log_artifact(fig_path)

# Mostrar la figura
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
