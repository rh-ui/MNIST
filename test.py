import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F


# Chargement des données
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

# Définition du modèle
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Instanciation du modèle
model = SimpleModel()

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entraînement du modèle
epochs = 5
for epoch in range(epochs):
    for data in trainset:
        X, y = data
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

# Évaluation du modèle
correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X, y = data
        output = model(X)
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

accuracy = correct / total
print(f'Précision du modèle sur l\'ensemble de test : {accuracy * 100:.2f}%')


# Ce code définit un modèle simple avec une seule couche linéaire (nn.Linear(28 * 28, 10)) pour représenter les 10 classes de chiffres de 0 à 9. Il utilise une fonction d'activation log_softmax pour générer des probabilités.

# Le modèle est ensuite entraîné sur l'ensemble de données MNIST pendant plusieurs époques, et sa précision est évaluée sur l'ensemble de test.