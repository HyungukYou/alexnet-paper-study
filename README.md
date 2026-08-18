# 🏛️ AlexNet Paper Reproduction & Experiment Suite

Google Colab 및 로컬 환경에서 **AlexNet (Krizhevsky et al., NIPS 2012)** 논문 원본 아키텍처 구현, Figure 3 흑백/컬러 분리 재현, Section 3.1 활성화 함수(ReLU vs Tanh vs Sigmoid) 성능 비교 실험을 수행할 수 있는 프로젝트입니다.

---

## 📁 프로젝트 구조

```
alexnet_project/
├── README.md                          # 프로젝트 설명 문서
├── main.py                            # 로컬 테스트 메인 실행 파일
├── alexnet_colab_notebook.ipynb       # Google Colab 원클릭 실행 노트북
├── models/
│   └── alexnet.py                     # OriginalAlexNet 및 TwoStreamAlexNet (Figure 3) 구현
└── experiments/
    └── act_comparison.py              # ReLU vs Tanh vs Sigmoid 수렴 속도 및 그래프 생성
```

---

## 🚀 빠른 시작 (Quick Start)

### 1. 로컬 환경 실행
```bash
python main.py
```

### 2. Google Colab 실행
`alexnet_colab_notebook.ipynb` 파일 내용을 Google Colab으로 업로드하거나 복사하여 실행합니다.

---

## 📊 주요 실험 및 시각화
1. **Figure 3 2-Stream 흑백/컬러 분리**:
   - GPU 1 스트림 (상단 48개 커널): 흑백 에지 및 고주파 텍스처 특화
   - GPU 2 스트림 (하단 48개 커널): 컬러 블롭 및 보색 패치 특화
2. **Section 3.1 활성화 함수 수렴 비교**:
   - ReLU vs Tanh vs Sigmoid 수렴 속도 비교 손실 하강 곡선 및 연산 소실 바 차트 생성
