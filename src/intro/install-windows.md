---
layout: docs
title: "🪟 Windows 파이토치 설치 가이드 (NVIDIA CUDA & CPU)"
permalink: /intro/install-windows.html
description: "Windows 10/11 환경에서 NVIDIA 그래픽 카드의 CUDA 가속 및 CPU 모드로 파이토치를 올바르게 설치하는 방법을 안내합니다."
keywords: "PyTorch Windows, PyTorch CUDA 설치, Windows pip torch, NVIDIA GPU 파이토치, jinydev"
---

# 🪟 Windows 파이토치 설치 가이드 (NVIDIA CUDA & CPU)

Windows 환경에서는 NVIDIA 외장 그래픽 카드(GeForce RTX 시리즈 등)의 유무에 따라 설치 명령어가 나뉩니다.

---

## 🔍 1단계: NVIDIA 그래픽 카드 및 드라이버 확인

1. **명령 프롬프트(CMD)** 또는 **PowerShell**을 엽니다.
2. 아래 명령어로 NVIDIA 드라이버가 인식되는지 확인합니다:
   ```cmd
   nvidia-smi
   ```
3. 표 우측 상단에 표시되는 **`CUDA Version: 12.x`** 숫자를 확인합니다. 이는 현재 드라이버가 지원할 수 있는 **최대 CUDA 버전**을 의미합니다.

---

## 📦 2단계: PyTorch 설치 명령어 실행

### 옵션 A: NVIDIA GPU (CUDA 12.1 권장)
최신 RTX 30xx, 40xx 그래픽 카드를 사용 중이라면 아래 명령어를 실행합니다:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 옵션 B: NVIDIA GPU (CUDA 11.8 구형 그래픽 카드)
GTX 10xx, 16xx 등 구형 그래픽 카드라면 CUDA 11.8 버전을 권장합니다:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 옵션 C: 외장 그래픽 카드가 없는 경우 (CPU Only)
내장 그래픽만 탑재된 노트북이나 사무용 PC라면 CPU 전용 버전을 설치합니다:

```cmd
pip install torch torchvision torchaudio
```

---

## 🚨 Windows 설치 시 자주 발생하는 문제 해결

### 1) Microsoft Visual C++ Redistributable 누락
파이토치 임포트 시 `DLL load failed while importing _C` 오류가 발생하면, [Microsoft 공식 웹사이트](https://learn.microsoft.com/ko-kr/cpp/windows/latest-supported-vc-redist)에서 **Visual C++ 2015-2022 재배포 가능 패키지(x64)**를 다운로드하여 설치해야 합니다.

### 2) CUDA 가속 확인
PowerShell에서 아래 코드로 GPU 인식을 검증합니다:

```cmd
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available(), '장치명:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')"
```
