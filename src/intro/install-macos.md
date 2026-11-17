---
layout: docs
title: "🍏 macOS 파이토치 설치 가이드 (Apple Silicon MPS 가속)"
permalink: /intro/install-macos.html
description: "Apple Silicon(M1, M2, M3, M4) 맥에서 Metal Performance Shaders(MPS) GPU 가속을 활용하는 파이토치 설치법을 안내합니다."
keywords: "PyTorch macOS, PyTorch Apple Silicon, PyTorch M1, PyTorch M2, PyTorch M3, MPS 가속, jinydev"
---

# 🍏 macOS 파이토치 설치 가이드 (Apple Silicon MPS 가속)

Apple Silicon(M1, M2, M3, M4 시리즈) 칩을 탑재한 Mac은 고성능 통합 메모리(Unified Memory) 구조를 갖추고 있어, 대용량 딥러닝 텐서 연산을 CPU와 GPU가 제로 카피(Zero-copy)로 공유할 수 있습니다.  
파이토치는 이를 위해 Apple의 **MPS(Metal Performance Shaders)** 백엔드를 공식 지원합니다.

---

## 🛠️ 단계별 설치 절차

### 1단계: 파이썬 및 가상환경 준비
macOS 패키지 관리자인 Homebrew를 통해 최신 파이썬 3.12를 설치하고 가상환경을 만듭니다:

```bash
# 1. 새 작업 폴더 생성
mkdir pytorch_lab && cd pytorch_lab

# 2. 가상환경 생성
python3 -m venv .venv

# 3. 가상환경 활성화
source .venv/bin/activate
```

### 2단계: PyTorch 공식 휠(Wheel) 설치
macOS용 공식 파이토치 패키지는 Apple Silicon arm64용 Metal 가속기를 기본 포함하고 있습니다:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

---

## 🍎 MPS(Metal) 하드웨어 가속 검증

설치가 끝난 후 아래 파이썬 코드를 실행하여 MPS 가속기가 인식되는지 확인합니다:

```python
import torch

print("PyTorch Version:", torch.__version__)

# Apple Silicon GPU 가속 가능 여부 확인
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("MPS 가속 가능 여부:", mps_available)

if mps_available:
    # MPS 가속 장치 생성
    device = torch.device("mps")
    
    # GPU 텐서 생성 및 행렬 곱셈 테스트
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.matmul(a, b)
    
    print("Apple Silicon GPU(MPS) 연산 성공! 행렬 크기:", c.shape)
else:
    print("MPS 가속기를 찾을 수 없습니다. Intel Mac이거나 macOS 12.3 미만입니다.")
```

{% include character_callout.html 
   char="dorothy" 
   title="맥북에서도 GPU 가속이 되네요!" 
   text="별도의 거대한 엔비디아 그래픽 카드 없이도 맥북 에어나 맥북 프로에서 딥러닝 모델 학습을 빠르게 돌릴 수 있다니 정말 신기해요!" 
%}
