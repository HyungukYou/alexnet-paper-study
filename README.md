# 🏛️ Deep Learning Paper Study & Implementation Suite (AlexNet & ResNet)

Google Colab 및 로컬 맥북 GPU(Apple Silicon MPS) 환경에서 대표적인 딥러닝 칭호 논문 아키텍처(**AlexNet**, **ResNet**)를 PyTorch로 직접 구현하고 실험하는 스터디 리포지토리입니다.

---

## 📌 논문 구현 및 스터디 스크립트 요약

### 1. ⚡ ResNet (He et al., CVPR 2016)
- **핵심 논문**: *Deep Residual Learning for Image Recognition*
- **핵심 문제 해결**: 신경망이 깊어질수록 발생하던 성능 저하(Degeneration Problem)와 기울기 소실(Vanishing Gradient) 문제를 **Residual Shortcut Connection** ($\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$)을 통해 완벽히 해결.

| 모듈 명칭 | 설명 및 아키텍처 스펙 | 관련 소스코드 |
| :--- | :--- | :--- |
| **BasicBlock** | ResNet-18 / ResNet-34 전용 2-Layer 잔차 블록 ($3 \times 3 \rightarrow 3 \times 3$) | `study_resnet.py` (`BasicBlock`) |
| **Bottleneck Block** | ResNet-50 / 101 / 152 전용 3-Layer 블록 ($1 \times 1 \rightarrow 3 \times 3 \rightarrow 1 \times 1$ 채널 4배 확장) | `study_resnet.py` (`Bottleneck`) |
| **Shortcut Connection** | Identity Shortcut & Projection Shortcut (1x1 Conv, stride 2) 자동 처리 | `study_resnet.py` |
| **Weight Init** | Kaiming (He) Normal Initialization 적용 (`kaiming_normal_`) | `study_resnet.py` (`_initialize_weights`) |
| **Unit Test** | ResNet-18/34/50 입출력 차원 및 역전파(Backpropagation) 자동 검증 | `test_resnet.py` |

---

### 2. 🏛️ AlexNet (Krizhevsky et al., NIPS 2012)
- **핵심 논문**: *ImageNet Classification with Deep Convolutional Neural Networks*
- **핵심 기능**: ReLU 활성화 함수, Local Response Normalization (LRN), Overlapping Pooling, Dropout (0.5), SGD Momentum (0.9).
- **관련 파일**: `study_alexnet.py`, `alexnet_colab_notebook.ipynb`

---

## 📁 프로젝트 파일 구조

```
alexnet-paper-study/
├── 📄 README.md                          # 논문 핵심 분석 및 실행 가이드
├── 🐍 study_resnet.py                    # ResNet-18/34/50 통합 구현 및 학습/시각화 스크립트
├── 🧪 test_resnet.py                     # ResNet-18/34/50 자동 단위 테스트 (Unit Tests)
├── 📊 resnet_study_result.png            # ResNet-18 학습 손실 수렴 시각화 그래프
├── 🐍 study_alexnet.py                   # AlexNet 올인원 공부 및 실습 스크립트
├── 📊 alexnet_study_result.png           # AlexNet 학습 손실 수렴 시각화 그래프
└── 📓 alexnet_colab_notebook.ipynb       # Google Colab 원클릭 실습 노트북
```

---

## 🚀 실행 가이드 (Apple Silicon GPU / CPU)

### 1. ResNet 구현 및 학습 테스트 실행
```bash
python study_resnet.py
```
- 맥북 Apple Silicon GPU (`mps`) 및 CPU 환경을 자동으로 감지합니다.
- CIFAR-10 데이터셋으로 ResNet-18 학습을 진행하고 `resnet_study_result.png` 그래프를 생성합니다.

### 2. ResNet 단위 테스트 (Unit Tests) 실행
```bash
python test_resnet.py
```
- ResNet-18, ResNet-34, ResNet-50의 forward/backward pass 및 tensor shape 호환성을 검증합니다.

### 3. AlexNet 학습 테스트 실행
```bash
python study_alexnet.py
```
