import torch
import torchvision
from torchvision import transforms, datasets

# tensor : a specialized data structure that are very similar to arrays and matrices.
train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

import matplotlib.pyplot as plt

plt.imshow(data[0][0].view(28, 28))
plt.show()
print(data[0][0].shape)

