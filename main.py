import os
import torch
import torchvision
import torchvision.transforms as transforms
from models.alexnet import TwoStreamAlexNet

def main():
    print("🚀 AlexNet Personal Project Main Script")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📌 Current Device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    
    model = TwoStreamAlexNet(num_classes=10).to(device)
    print("✅ TwoStreamAlexNet (Figure 3 Paper Model) initialized successfully!")
    print(model)

if __name__ == "__main__":
    main()
