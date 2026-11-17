---
layout: docs
title: "4. 최신 파이토치 버전과 PyTorch 2.x"
permalink: /intro/versions.html
description: "torch.compile 컴파일러 도입, TorchDynamo, TorchInductor, Python 3.12 지원 등 PyTorch 2.x의 최신 진화 사항을 정리합니다."
keywords: "PyTorch 2.0, torch.compile, TorchDynamo, TorchInductor, Triton, PyTorch 최신 버전, jinydev"
---

# 4. 최신 파이토치 버전과 PyTorch 2.x의 혁신

과거 파이토치 1.x 시리즈가 '사용자 편의성과 동적 계산 그래프의 유연함'에 집중했다면, **PyTorch 2.0 이후 세대**는 **"파이썬다운 유연성을 그대로 유지하면서도 C++ 수준의 극한 속도를 달성하는 컴파일러 혁신"**을 이룩했습니다.

---

## ⚡ 1. `torch.compile()`: 단 한 줄의 가속 혁명

PyTorch 2.0의 가장 핵심적인 기능은 `torch.compile()` 함수입니다:

```python
import torch

# 기존 신경망 모델 정의
model = MyDeepResNet().to("cuda")

# 🚀 컴파일 활성화: 단 한 줄로 30%~100% 가속!
opt_model = torch.compile(model)

# 기존과 완전히 동일하게 실행
outputs = opt_model(inputs)
```

이전 1.x 시절의 TorchScript는 엄격한 타입 규칙과 파이썬 서브셋 문법만을 요구하여 디버깅이 매우 까다로웠으나, `torch.compile`은 100% 순수 파이썬 코드를 있는 그대로 지원합니다.

---

## 🏗️ 2. PyTorch 2.x를 지탱하는 4대 기술 스택

```text
┌─────────────────────────────────────────────────────────────┐
│                      사용자 파이썬 코드                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. TorchDynamo: 파이썬 바이트코드 수준에서 그래프 안전하게 캡처 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. AOTAutograd: 순전파뿐만 아니라 역전파 그래프까지 사전 캡처 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. PrimTorch: 수천 개의 파이토치 연산자를 250개 코어로 단순화 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TorchInductor: OpenAI Triton을 사용한 GPU 고속 커널 생성  │
└─────────────────────────────────────────────────────────────┘
```

{% include character_callout.html 
   char="tinny" 
   title="Triton GPU 가속 엔진" 
   text="OpenAI가 개발한 Triton 언어를 백엔드로 사용하여 엔비디아 CUDA C++ 코드를 직접 짜지 않고도 메모리 대역폭을 극대화하는 퓨전 커널(Kernel Fusion)을 자동 생성합니다!" 
%}

---

## 📅 주요 버전별 핵심 변화

| 버전 | 출시 시점 | 핵심 신기능 |
| :---: | :---: | :--- |
| **PyTorch 2.5** | 2024년 말 | CUDAGraph 자동화 개선, FlexAttention 고속 어텐션 API, Python 3.12 지원 안정화 |
| **PyTorch 2.0** | 2023년 봄 | `torch.compile()` 최초 도입, TorchDynamo, TorchInductor 정식 릴리스 |
| **PyTorch 1.13**| 2022년 말 | Apple Silicon MPS(Metal) 가속 정식 지원 |
| **PyTorch 1.0** | 2018년 말 | Caffe2 통합, TorchScript 도입 |
