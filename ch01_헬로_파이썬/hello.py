"""
===================================================================
📘 밑바닥부터 시작하는 딥러닝 - 1장: 헬로 파이썬 (Hello Python)
===================================================================
1. 기본 산술 연산 및 변수
2. 넘파이(NumPy) 배열 연산 & 행렬 곱
3. 맷플롯립(Matplotlib) 삼각함수(Sin/Cos) 그래프 시각화
"""

import numpy as np
import matplotlib.pyplot as plt

print("🚀 [ch01 헬로 파이썬] 실습 스크립트 시작\n")

# 1. 기본 산술 연산
a = 10
b = 20
print(f"1. 기본 연산: {a} + {b} = {a + b}")

# 2. 넘파이(NumPy) N차원 배열 생성 및 방송(Broadcasting)
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

print(f"\n2. 넘파이 배열 연산:")
print(f"   x = {x}")
print(f"   y = {y}")
print(f"   x + y = {x + y}")
print(f"   x * y = {x * y}")

# 2D 행렬 곱 (Dot Product)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\n3. 2D 행렬 곱 (A @ B):\n{np.dot(A, B)}")

# 3. 맷플롯립(Matplotlib) 그래프 시각화
print("\n📊 Sin & Cos 함수 그래프 생성 중...")
x_vals = np.arange(0, 6, 0.1) # 0부터 6까지 0.1 간격
y_sin = np.sin(x_vals)
y_cos = np.cos(x_vals)

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_sin, label="sin(x)", color="blue", linewidth=2)
plt.plot(x_vals, y_cos, label="cos(x)", color="red", linestyle="--", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Sin & Cos Wave Plot (Ch01 Practice)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("ch01_sin_cos_plot.png", dpi=200)
print("✅ 'ch01_sin_cos_plot.png' 그래프 저장 완료!")

print("\n🎉 [ch01 헬로 파이썬] 모든 실습 코드가 성공적으로 실행되었습니다!")
