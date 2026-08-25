"""
===================================================================
📚 ResNet 논문(He et al., CVPR 2016) 올인원(All-In-One) 학습 스크립트
===================================================================
Deep Residual Learning for Image Recognition 논문의 핵심 아이디어:
1. Residual Learning (잔차 학습): F(x) = H(x) - x 를 학습하여 F(x) + x 를 복원
2. Identity Shortcut Mapping: 추가 파라미터나 계산량 없이 기울기 소실(Vanishing Gradient) 문제 해결
3. BasicBlock (ResNet-18/34) & Bottleneck Block (ResNet-50/101/152) 
4. Kaiming (He) Normal Initialization: ReLU 활성화 함수에 최적화된 가중치 초기화

이 스크립트는 ResNet-18, ResNet-34, ResNet-50 아키텍처 구현 및 CIFAR-10 데이터셋 실습을 포함합니다.
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
# 1단계: Residual Block 모듈 정의 (BasicBlock & Bottleneck)
# =================================================================

class BasicBlock(nn.Module):
    """
    [ResNet-18 / ResNet-34 용 2-Layer Residual Block]
    - Conv1 (3x3, stride=s) -> BN -> ReLU
    - Conv2 (3x3, stride=1) -> BN
    - Shortcut: stride != 1 이거나 채널 수가 다를 경우 1x1 Conv (Projection Shortcut)
    - Output: ReLU(F(x) + shortcut(x))
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut Connection (Identity or Projection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual Connection (Element-wise Addition)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    [ResNet-50 / ResNet-101 / ResNet-152 용 3-Layer Bottleneck Block]
    - Conv1 (1x1, channels) -> BN -> ReLU (차원 축소)
    - Conv2 (3x3, channels, stride=s) -> BN -> ReLU (연산 수행)
    - Conv3 (1x1, channels * 4) -> BN (차원 확장)
    - Shortcut: stride != 1 이거나 in_channels != out_channels * 4 일 때 1x1 Projection
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        # 1x1 Conv: 입력 채널 -> out_channels 축소
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 Conv: 공간 연산 (stride 적용)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 Conv: out_channels -> out_channels * 4 확장
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Shortcut Connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # Residual Connection
        out = F.relu(out)
        return out


# =================================================================
# 2단계: ResNet 메인 아키텍처 클래스 정의
# =================================================================

class ResNet(nn.Module):
    """
    ResNet 메인 아키텍처 (Paper Spec)
    - Stem Layer: Conv (7x7, 64, stride 2, padding 3) -> BN -> ReLU -> MaxPool (3x3, stride 2)
    - Layer 1 (conv2_x): 64 channels
    - Layer 2 (conv3_x): 128 channels (stride 2)
    - Layer 3 (conv4_x): 256 channels (stride 2)
    - Layer 4 (conv5_x): 512 channels (stride 2)
    - Head: AdaptiveAvgPool (1x1) -> Linear(512 * expansion, num_classes)
    """
    def __init__(self, block, num_blocks, num_classes=10, cifar_stem=False):
        super(ResNet, self).__init__()
        self.in_channels = 64

        if cifar_stem:
            # CIFAR-10 (32x32) 해상도 맞춤형 Stem
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.Identity()
        else:
            # ImageNet (224x224) 원본 Stem
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # He (Kaiming) Normal Initialization
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 편의용 모델 생성 함수
def ResNet18(num_classes=10, cifar_stem=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, cifar_stem=cifar_stem)

def ResNet34(num_classes=10, cifar_stem=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, cifar_stem=cifar_stem)

def ResNet50(num_classes=10, cifar_stem=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, cifar_stem=cifar_stem)


# =================================================================
# 3단계: 메인 학습 및 결과 시각화 실행 (Main Program)
# =================================================================

def main():
    print("🚀 [ResNet 공부용] All-In-One 단일 파일 학습 스크립트 (He et al., 2016)")
    
    # 디바이스 자동 선택 (맥북 Apple Silicon GPU: mps / CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("⚡ [Device] 맥북 Apple Silicon GPU (MPS) 가속기 활성화!")
    else:
        device = torch.device('cpu')
        print("💻 [Device] CPU 가속기 활성화!")

    # 데이터 전처리 (ImageNet 224x224 해상도 리사이징 적용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("📦 CIFAR-10 실제 데이터셋 로드 중...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # 모델 생성 (ResNet-18)
    model = ResNet18(num_classes=10, cifar_stem=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    print("\n🏋️ 맥북 GPU 기반 ResNet-18 학습 진행 중...")
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
        if i >= 25:  # 수렴 확인용 25 Step 실습
            break

    elapsed = time.time() - start_time
    print(f"\n🎉 학습 완료! 소요 시간: {elapsed:.2f}초")

    # =================================================================
    # 4단계: 학습 손실(Loss) 수렴 그래프 시각화 출력
    # =================================================================
    plt.figure(figsize=(9, 4.5))
    plt.plot(losses, marker='s', color='royalblue', linewidth=2.5, label='ResNet-18 Training Loss')
    plt.title('ResNet-18 Training Loss Convergence (MacBook GPU Accelerated)', fontsize=13, fontweight='bold')
    plt.xlabel('Training Steps (x5 iterations)', fontsize=11)
    plt.ylabel('Cross Entropy Loss', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('resnet_study_result.png', dpi=200)
    print("📊 학습 손실 시각화 그래프가 'resnet_study_result.png' 파일로 저장되었습니다!")
    plt.show()

if __name__ == "__main__":
    main()
