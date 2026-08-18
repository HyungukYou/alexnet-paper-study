"""
Main Execution Script for AlexNet Paper Reproduction & Acceleration Test.
Optimized for Apple Silicon Mac GPU (MPS) & PyTorch.
"""
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from models.alexnet import TwoStreamAlexNet, OriginalAlexNet


def get_device() -> torch.device:
    """Select the best available compute device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        print("⚡ [Device] Apple Silicon Mac GPU (MPS) Accelerated!")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("⚡ [Device] CUDA GPU Accelerated!")
        return torch.device("cuda")
    else:
        print("💻 [Device] CPU Mode")
        return torch.device("cpu")


def main():
    print("🚀 [AlexNet Project] Main Model Execution Pipeline")
    device = get_device()

    # ImageNet Standard Normalization & Resizing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    print("📦 Loading CIFAR-10 Dataset...")
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize AlexNet TwoStream Model (Figure 3 Paper Specialization)
    model = TwoStreamAlexNet(num_classes=10).to(device)
    print("✅ TwoStreamAlexNet (Paper Figure 3 Model) Ready!")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    print("\n🏋️ Training Test Cycle Started...")
    start_time = time.time()
    model.train()

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49:
            print(f"Batch [{i+1}/{len(trainloader)}], Loss: {running_loss / 50:.4f}")
            running_loss = 0.0
        if i >= 100:  # Fast validation loop
            break

    elapsed = time.time() - start_time
    print(f"🎉 Training Test Completed! Elapsed Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
