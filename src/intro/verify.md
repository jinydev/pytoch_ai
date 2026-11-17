---
layout: docs
title: "6. 환경 검증 및 첫 텐서 테스트"
permalink: /intro/verify.html
description: "CPU, NVIDIA CUDA, Apple Silicon MPS 등 하드웨어 가속기 상태를 점검하고 기본 텐서 연산을 수행하는 통합 검증 스크립트를 제공합니다."
keywords: "PyTorch 검증, PyTorch 테스트, torch.cuda.is_available, MPS 테스트, jinydev"
---

# 6. 환경 검증 및 첫 텐서 테스트

파이토치 설치를 완료했다면, 아래 통합 점검 스크립트를 작성하여 컴퓨터의 가속 엔진이 정상 동작하는지 테스트합니다.

---

## 🧪 통합 하드웨어 점검 파이썬 스크립트

아래 코드를 복사하여 `check_env.py` 파일로 저장하거나 인터프리터에서 실행하세요:

```python
import sys
import torch

print("=" * 60)
print(f"🐍 Python Executable : {sys.executable}")
print(f"🐍 Python Version    : {sys.version.split()[0]}")
print(f"🔥 PyTorch Version   : {torch.__version__}")
print("=" * 60)

# 가속 장치 판별
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    cuda_ver = torch.version.cuda
    print(f"🚀 [가속 모드] NVIDIA CUDA 활성화")
    print(f"   - GPU 장치명 : {device_name} (총 {device_count}대)")
    print(f"   - CUDA 버전  : {cuda_ver}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 [가속 모드] Apple Silicon Metal(MPS) 활성화")
else:
    device = torch.device("cpu")
    print("💻 [일반 모드] CPU 가속 (GPU 가속 장치 없음)")

print("-" * 60)

# 텐서 생성 및 행렬 곱 연산 벤치마크 테스트
print(f"⚡ {device} 장치에서 2000x2000 행렬 곱 연산 테스트 중...")
x = torch.randn(2000, 2000, device=device)
y = torch.randn(2000, 2000, device=device)
z = torch.matmul(x, y)

print(f"✅ 연산 성공! 결과 텐서 형태: {z.shape}, 장치: {z.device}")
print("=" * 60)
print("🎉 모든 환경 설정이 성공적으로 완료되었습니다!")
```

---

## 🧭 다음 학습 단계

모든 환경 검증이 완료되었습니다! 이제 파이토치와 함께 딥러닝 여정을 본격적으로 시작합니다:

- **[01장. 딥러닝 시작](../01_딥러닝_시작/)**: 인공지능과 머신러닝, 딥러닝의 역사적 흐름
- **[02장. 파이토치를 위한 파이썬 선수 지식](../02_파이썬/)**: 파이썬 마스터 클래스 연동 및 넘파이 텐서 비교
- **[04장. 파이토치 기본](../04_파이토치_기본/)**: 텐서 조작과 Autograd 자동 미분
