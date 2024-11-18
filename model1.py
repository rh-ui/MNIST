import torch
import torchvision
from torchvision import transforms, datasets


# tensor : a specialized data structure that are very similar to arrays and matrices.
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))  #Telecharge l'ensemble de données MNIST d'apprentissage, le charge et applique une transformation pour convertir les images en tenseurs.

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True) #Crée un DataLoader pour l'ensemble de données d'apprentissage avec une taille de lot de 10 et mélange les donnees
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): #Definit une classe appelee Net qui herite de nn.Module, la classe de base pour tous les modules PyTorch
    def __init__(self):
        super().__init__()
        #define our layers :
                            #input  output
        self.fc1 = nn.Linear(28 * 28, 64) # 28 * 28 : number of pxls
        self.fc2 = nn.Linear(64, 64) # we can put whatever we want in the output but the input should be the last output
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) # 10 cause we have 10 classes

    def forward(self, x): #definit le passage forward a travers le réseau
        x = F.relu(self.fc1(x))
        #   ^--> Activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.log_softmax(x, dim=1) # Applique une fonction log_softmax a la sortie de la derniere couche, fournissant la sortie du reseau sous forme de distribution de probabilites.
        #dim =1 means which thing is the probability distribution that we want to sum to one


net = Net()  # Instanciation du modele

X = torch.rand((28, 28))  # Creation d'un random tensor de dimensions 28x28
X = X.view(-1, 28 * 28)   # Remodelage du tensor pour correspondre a la taille d'entrée du réseau
output = net(X)           # Passage forward a travers le reseau
# print(output)

import torch.optim as optim   #we optimize for the loss

optimizer = optim.Adam(net.parameters(), lr=0.001) #lr: learning rate 

EPOCHS = 3 # ephoc = full pass through data (epoque)

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of features & labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28 * 28))
        loss = F.nll_loss(output, y) 
        loss.backward()
        optimizer.step()
    # print(loss)

correct = 0
total = 0
with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 28 * 28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
            
# print("Accuracy : ", round(correct/total, 3))

import matplotlib.pyplot as plt
plt.imshow(X[1].view(28, 28))
plt.show()

print(torch.argmax(net(X[1].view(-1,784))))





