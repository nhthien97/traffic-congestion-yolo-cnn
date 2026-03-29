import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# transform
transform = transforms.Compose([
    transforms.Resize((100,100)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# dataset
dataset = ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# model CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.Conv2d(32,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*48*48,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
for epoch in range(5):
    for imgs, labels in loader:
        labels = labels.float().unsqueeze(1)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# lưu model
torch.save(model.state_dict(), "models/cnn.pth")

print("Training done & model saved!")