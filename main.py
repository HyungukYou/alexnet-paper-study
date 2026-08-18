"""
Synthetic Data Test Script for AlexNet (No Network Download Required)
Performs immediate local GPU acceleration testing using standard tensors.
"""
import time
import torch
import torch.nn as nn
from models.alexnet import TwoStreamAlexNet


def get_device() -> torch.device:
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
    print("🚀 [AlexNet Project] Fast Synthetic GPU Execution Pipeline")
    device = get_device()

    # Create Synthetic Input Batch (Batch=32, Channels=3, H=224, W=224)
    dummy_input = torch.randn(32, 3, 224, 224, device=device)
    dummy_target = torch.randint(0, 10, (32,), device=device)

    # Initialize AlexNet TwoStream Model (Figure 3 Paper Specialization)
    model = TwoStreamAlexNet(num_classes=10).to(device)
    print("✅ TwoStreamAlexNet (Paper Figure 3 Model) Ready on GPU!")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("\n🏋️ Running Fast Synthetic GPU Training Test (5 Steps)...")
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
    print(f"\n🎉 Synthetic GPU Test Completed! Total Time: {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
