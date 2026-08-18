"""
Main Execution Pipeline for AlexNet Paper Reproduction.
Optimized for Apple Silicon Mac GPU (MPS) Acceleration.
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
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


def run_pipeline(use_synthetic: bool = True):
    """
    Run AlexNet Training Cycle.
    :param use_synthetic: If True, uses synthetic tensors for instant GPU verification.
    """
    print("🚀 [AlexNet Project] Main Execution Pipeline")
    device = get_device()

    # Initialize AlexNet TwoStream Model (Section 3.2 Figure 3)
    model = TwoStreamAlexNet(num_classes=10).to(device)
    print("✅ TwoStreamAlexNet (Paper Figure 3 Model) Ready!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    if use_synthetic:
        print("\n🏋️ [Synthetic Mode] Running 5-step Instant GPU Verification...")
        dummy_input = torch.randn(32, 3, 224, 224, device=device)
        dummy_target = torch.randint(0, 10, (32,), device=device)

        start_time = time.time()
        model.train()
        for step in range(1, 6):
            optimizer.zero_grad()
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_target)
            loss.backward()
            optimizer.step()
            print(f"  Step [{step}/5] Loss: {loss.item():.4f}")

        elapsed = time.time() - start_time
        print(f"\n🎉 Synthetic GPU Test Completed! Total Time: {elapsed:.2f} seconds")

    else:
        print("\n📦 [Dataset Mode] Loading CIFAR-10 Dataset...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        data_dir = "./data"
        os.makedirs(data_dir, exist_ok=True)

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

        print("🏋️ [CIFAR-10 Mode] Training AlexNet on Mac GPU...")
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
            if i % 20 == 19:
                print(f"  Batch [{i+1}/{len(trainloader)}], Loss: {running_loss / 20:.4f}")
                running_loss = 0.0
            if i >= 60:  # Fast validation cycle
                break

        elapsed = time.time() - start_time
        print(f"\n🎉 CIFAR-10 GPU Training Cycle Completed! Total Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    run_pipeline(use_synthetic=True)
