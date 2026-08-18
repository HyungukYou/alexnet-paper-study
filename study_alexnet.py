"""
===================================================================
📚 AlexNet 논문(2012) 올인원(All-In-One) 한 줄 통합 공부 스크립트
===================================================================
학교 서버의 Jupyter Notebook처럼 [1. 모델 구조] -> [2. 데이터 로드] -> 
[3. 학습 및 검증] -> [4. 결과 그래프 시각화]가 이 파일 하나에 전부 들어있습니다!
위에서부터 순서대로 읽으시면 AlexNet의 모든 작동 원리를 한눈에 파악하실 수 있습니다.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# =================================================================
# 1단계: AlexNet 신경망 모델 정의 (Krizhevsky et al., NIPS 2012)
# =================================================================

class LocalResponseNorm(nn.Module):
    """
    [Section 3.3] Local Response Normalization (LRN)
    논문 공식: k=2, n=5, alpha=1e-4, beta=0.75
    """
    def __init__(self, size=5, alpha=1e-4, beta=0.75, k=2.0):
        super(LocalResponseNorm, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if x.device.type == 'mps':
            return F.local_response_norm(x.cpu(), self.size, self.alpha, self.beta, self.k).to(x.device)
        return F.local_response_norm(x, self.size, self.alpha, self.beta, self.k)


class AlexNetStudyModel(nn.Module):
    """
    [Section 3.2 & 3.5] AlexNet 8-Layer 전체 구조 (Figure 3 복원)
    - Conv1 (96필터, 11x11, Stride 4) -> LRN -> Overlapping Pool (3x3, Stride 2)
    - Conv2 (256필터, 5x5, Padding 2) -> LRN -> Overlapping Pool (3x3, Stride 2)
    - Conv3 (384필터, 3x3)
    - Conv4 (384필터, 3x3)
    - Conv5 (256필터, 3x3) -> Overlapping Pool (3x3, Stride 2)
    - FC6 (4096) + Dropout (0.5)
    - FC7 (4096) + Dropout (0.5)
    - FC8 (10 classes)
    """
    def __init__(self, num_classes=10):
        super(AlexNetStudyModel, self).__init__()
        self.lrn = LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)
        
        # 1-2번 레이어: Conv + LRN + Overlapping Pooling (3x3, stride 2)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 3-5번 레이어: 3차원 Conv 구조
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # 6-8번 레이어: FC Fully-Connected Classifier + Dropout (0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        # Conv Layer 1 & 2 (with LRN & Overlapping Pooling)
        x = self.pool1(self.lrn(F.relu(self.conv1(x))))
        x = self.pool2(self.lrn(F.relu(self.conv2(x))))
        
        # Conv Layer 3, 4, 5
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(F.relu(self.conv5(x)))
        
        # Classifier FC Layer 6, 7, 8
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =================================================================
# 2단계: 메인 학습 및 결과 시각화 실행 (Main Program)
# =================================================================

def main():
    print("🚀 [AlexNet 공부용] All-In-One 단일 파일 학습 스크립트")
    
    # 디바이스 자동 선택 (맥북 Apple Silicon GPU: mps)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("⚡ [Device] 맥북 Apple Silicon GPU (MPS) 가속기 활성화!")
    else:
        device = torch.device('cpu')
        print("💻 [Device] CPU 가속기 활성화!")

    # 데이터 전처리 (ImageNet 224x224 리사이징 & 정규화)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("📦 CIFAR-10 실제 데이터셋 로드 중...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    # 모델 & 손실함수 & 옵티마이저 생성 (Section 5 SGD Momentum 0.9)
    model = AlexNetStudyModel(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    print("\n🏋️ 맥북 GPU 기반 AlexNet 학습 진행 중...")
    start_time = time.time()
    model.train()
    
    losses = []
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 5 == 4:
            avg_loss = running_loss / 5
            losses.append(avg_loss)
            print(f"  Step [{i+1}/{len(trainloader)}], Loss: {avg_loss:.4f}")
            running_loss = 0.0
        if i >= 25:  # 학습 수렴 확인용 25 Step 실습
            break

    elapsed = time.time() - start_time
    print(f"\n🎉 학습 완료! 소요 시간: {elapsed:.2f}초")

    # =================================================================
    # 3단계: 학습 손실(Loss) 수렴 그래프 시각화 출력
    # =================================================================
    plt.figure(figsize=(9, 4.5))
    plt.plot(losses, marker='o', color='crimson', linewidth=2.5, label='AlexNet Training Loss')
    plt.title('AlexNet Training Loss Convergence (MacBook GPU Accelerated)', fontsize=13, fontweight='bold')
    plt.xlabel('Training Steps (x5 iterations)', fontsize=11)
    plt.ylabel('Cross Entropy Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('alexnet_study_result.png', dpi=200)
    print("📊 학습 손실 시각화 그래프가 'alexnet_study_result.png' 파일로 저장되었습니다!")
    plt.show()

if __name__ == "__main__":
    main()
