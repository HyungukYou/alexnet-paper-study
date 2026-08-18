# 🏛️ AlexNet Paper (Krizhevsky et al., NIPS 2012) Full Reproduction & Experiment Suite

Google Colab 및 로컬 맥북 GPU(Apple Silicon MPS) 환경에서 **AlexNet (Krizhevsky et al., NIPS 2012)** 논문 원본 아키텍처 및 7가지 핵심 기능을 완벽히 재현한 프로젝트입니다.

---

## 📌 논문(2012) 핵심 기능 구현 현황

| 논문 Section | 논문 명칭 & 핵심 기능 | 구현 파일 / 위치 |
| :--- | :--- | :--- |
| **Section 3.1** | **ReLU Nonlinearity** ($f(x) = \max(0, x)$ 도입 및 Tanh/Sigmoid 대비 빠른 수렴) | `models/alexnet.py`, `experiments/act_comparison.py` |
| **Section 3.2** | **Two-GPU Parallel Stream Architecture** (GPU 1 흑백 에지 vs GPU 2 컬러 블롭 분리) | `models/alexnet.py` (`TwoStreamAlexNet`) |
| **Section 3.3** | **Local Response Normalization (LRN)** ($k=2, n=5, \alpha=10^{-4}, \beta=0.75$) | `models/alexnet.py` (`LocalResponseNorm`) |
| **Section 3.4** | **Overlapping Max Pooling** ($3 \times 3$ 커널, Stride $2$, 중첩 풀링) | `models/alexnet.py` (`MaxPool2d`) |
| **Section 3.5** | **Overall Architecture** (Conv1~5, FC6~8 8-Layer 구조) | `models/alexnet.py` |
| **Section 4.1** | **Data Augmentation** (Random Cropping, Horizontal Flipping, Normalization) | `main.py` |
| **Section 4.2** | **Dropout (p=0.5)** (FC6, FC7 과적합 방지) | `models/alexnet.py` |
| **Section 5** | **SGD with Momentum (0.9) & Weight Decay (0.0005)** | `main.py`, `experiments/act_comparison.py` |

---

## 📁 프로젝트 파일 구조

```
alexnet_project/
├── 📄 README.md                          # 논문 기능 정리 및 가이드
├── 🐍 main.py                            # 메인 파이프라인 (맥북 GPU MPS 가속)
├── 📓 alexnet_colab_notebook.ipynb       # Google Colab 원클릭 실행 노트북
├── 📁 models/
│   └── 🐍 alexnet.py                     # OriginalAlexNet 및 TwoStreamAlexNet (Paper Specs)
└── 📁 experiments/
    └── 🐍 act_comparison.py              # Section 3.1 활성화 함수 수렴 비교 모듈
```

---

## 🚀 실행 가이드

### 맥북 로컬 실행 (Apple Silicon GPU)
```bash
python main.py
```
