"""
AlexNet Main Execution Script
Zero Disk Space Usage (Memory Tensor Mode) & Apple Silicon GPU Acceleration
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models.alexnet import TwoStreamAlexNet


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
    """
    Main AlexNet Training Pipeline using 0MB Disk Synthetic Tensors.
    """
    print("🚀 [AlexNet Project] Zero-Disk Memory Tensor GPU Pipeline")
    device = get_device()

    # Initialize AlexNet TwoStream Model (Section 3.2 Figure 3)
    model = TwoStreamAlexNet(num_classes=10).to(device)
    print("✅ TwoStreamAlexNet (Paper Figure 3 Model) Ready!")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    print("\n🏋️ [In-Memory Mode] Running 5-step Instant GPU Training (0 MB Disk Usage)...")
    
    # 0MB Disk Space: Generated directly in RAM/GPU Memory
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
    print(f"\n🎉 GPU Training Completed! Total Time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
