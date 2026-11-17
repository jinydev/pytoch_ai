---
layout: docs
title: "파이토치(PyTorch) 소개 & 시작하기 가이드"
permalink: /intro/
description: "파이토치 프레임워크의 철학과 특징, 탄생 배경 및 역사, 공식 로고의 상징성, 최신 PyTorch 2.x 기능과 운영체제별(macOS, Windows, Linux) 설치 방법을 안내합니다."
keywords: "파이토치, PyTorch, 파이토치 소개, 파이토치 설치, 파이토치 로고, PyTorch CUDA, PyTorch MPS, 딥러닝, jinydev"
---

# 🔥 파이토치(PyTorch) 프레임워크 소개 & 시작하기

{% include lecture_illustration.html 
   src="/assets/img/logos/logo_09_pytorch.jpg" 
   alt="도로시와 함께하는 파이토치 대탐험" 
   caption="딥러닝 연구와 산업계를 이끄는 가장 우아하고 강력한 프레임워크 — 파이토치(PyTorch)의 세계에 오신 것을 환영합니다!" 
   character="도로시 탐험대 공식 마스터 삽화" 
%}

> *"From research to production, an open source machine learning framework that accelerates the path."*  
> *(연구에서 상용 서비스까지, 인공지능 개발의 여정을 가속하는 오픈소스 머신러닝 프레임워크)*  
> — PyTorch 공식 슬로건

**파이토치(PyTorch)**는 현재 전 세계 인공지능(AI) 연구원들과 머신러닝 엔지니어들이 가장 널리 사용하는 표준 딥러닝 프레임워크입니다.  
본 가이드에서는 파이토치의 철학과 역사, 공식 로고의 상징성, 최신 PyTorch 2.x의 혁신적 진화, 그리고 여러분의 컴퓨터(macOS, Windows, Linux)에 올바르게 설치하고 하드웨어 가속(CUDA/MPS)을 활성화하는 방법을 도로시 탐험대와 함께 상세히 안내합니다.

{% include character_callout.html 
   char="dorothy" 
   title="파이토치로 딥러닝 인공지능을 직접 만들 수 있나요?" 
   text="선생님! 이미지 분류 모델, 챗봇 언어 모델, 로봇 비전까지 전부 파이토치로 만들 수 있다면서요? 텐서플로우와는 무엇이 다른가요?" 
%}

{% include character_callout.html 
   char="hojin" 
   title="파이토치의 본질: 연구자의 생각대로 코딩되는 직관성" 
   text="도로시, 파이토치는 '동적 계산 그래프(Define-by-Run)' 방식을 사용하여 일반 파이썬 코드를 작성하듯 자연스럽게 신경망을 만들고 한 줄씩 디버깅할 수 있는 가장 파이썬다운(Pythonic) 딥러닝 프레임워크란다." 
%}

---

## 1. 파이토치 핵심 특징과 철학

### 🌟 4대 핵심 강점

1. **동적 계산 그래프 (Define-by-Run)**:  
   정적인 그래프 구조를 먼저 컴파일하고 데이터를 흘려보내는 기존 방식과 달리, 데이터가 입력되어 순전파 연산이 일어나는 순간에 실시간으로 연산 그래프를 구성합니다. 조건문(`if`), 반복문(`for`)을 신경망 내부에서 자유롭게 사용할 수 있습니다.
2. **파이썬다운 직관적 문법 (Pythonic & Readable)**:  
   파이썬의 표준 수치 라이브러리인 넘파이(NumPy)와 거의 1:1로 대응되는 API 구조를 제공하며, `print()`나 `pdb`와 같은 일반 파이썬 디버거를 그대로 사용할 수 있습니다.
3. **강력한 자동 미분 엔진 (Autograd Engine)**:  
   `requires_grad=True` 설정 하나만으로 복잡한 다변수 합성 함수의 연쇄 법칙(Chain Rule) 역전파 미분을 오차 없이 자동으로 계산합니다.
4. **강력한 하드웨어 가속 지원 (CUDA & Apple Silicon MPS)**:  
   코드 한 줄(`tensor.to('cuda')` 또는 `tensor.to('mps')`)로 CPU 연산을 수천 개의 코어를 가진 GPU 가속 장치로 즉시 전환할 수 있습니다.

{% include character_callout.html 
   char="jiny" 
   title="NumPy와 PyTorch 텐서는 쌍둥이다냥!" 
   text="NumPy의 다차원 배열 ndarray와 PyTorch Tensor는 구조가 거의 동일하다냥! 다만 텐서는 GPU에서 고속 연산이 가능하고 역전파 미분(Autograd)을 자동으로 추적한다는 엄청난 차이가 있다냥!" 
%}

---

## 2. 탄생과 발전 배경 (History)

{% include lecture_illustration.html 
   src="/assets/img/hojin_lab_lecture.jpg" 
   alt="이호진 교수님의 파이토치 탄생사 세미나" 
   caption="Lua 기반의 Torch에서 현대 딥러닝의 표준 PyTorch가 되기까지 — 이호진 교수님의 연구실 세미나" 
   character="이호진 교수님" 
%}

<div class="row align-items-center my-4 p-4 rounded-4 bg-light border">
  <div class="col-md-8">
    <h4 class="fw-bold text-dark mb-2">🏛️ 페이스북 AI 연구소(FAIR)에서 리눅스 재단(Linux Foundation)으로</h4>
    <p class="text-secondary small mb-2">
      파이토치는 2016년, 당시 페이스북(현 메타) AI 연구소(FAIR)의 <strong>아담 파스케(Adam Paszke)</strong>와 <strong>수미스 친탈라(Soumith Chintala)</strong> 등이 주도하여 기존 Lua 언어 기반의 'Torch' 프레임워크를 파이썬 생태계에 맞춰 전면 재설계하며 탄생했습니다.
    </p>
    <p class="text-secondary small mb-0">
      2017년 1월 오픈소스로 공개된 이후, 학계(NeurIPS, ICML, CVPR) 논문 코드의 80% 이상을 점유하며 세계 1위 프레임워크로 확고히 자리잡았습니다. 2022년 9월에는 메타에서 <strong>리눅스 재단(Linux Foundation)</strong> 산하의 독립 기구인 <strong>PyTorch Foundation</strong>으로 거버넌스가 공식 이관되어 중립적인 글로벌 오픈소스 프로젝트로 발전하고 있습니다.
    </p>
  </div>
  <div class="col-md-4 text-center">
    <span class="badge rounded-pill bg-danger px-3 py-2 fw-bold fs-6">PyTorch Foundation</span>
    <p class="small text-muted mt-2 mb-0">Linux Foundation 산하<br>중립 거버넌스 운영</p>
  </div>
</div>

---

## 3. 공식 로고 및 상징 (Logo & Brand Identity)

<div class="row align-items-center my-4 p-4 rounded-4 bg-white border shadow-sm">
  <div class="col-md-4 text-center mb-3 mb-md-0">
    <div class="p-3 d-inline-block rounded-circle bg-light border">
      <svg width="110" height="110" viewBox="0 0 160 160" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M79.5 20C46.6 20 20 46.6 20 79.5C20 112.4 46.6 139 79.5 139C112.4 139 139 112.4 139 79.5C139 63.3 132.6 48.7 122.2 38" stroke="#EE4C2C" stroke-width="12" stroke-linecap="round"/>
        <path d="M106 20L92 34L106 48" stroke="#EE4C2C" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M79.5 50C79.5 50 63 67 63 80C63 89.1 70.4 96.5 79.5 96.5C88.6 96.5 96 89.1 96 80C96 67 79.5 50 79.5 50Z" fill="#EE4C2C"/>
        <circle cx="94" cy="65" r="4.5" fill="#EE4C2C"/>
      </svg>
    </div>
    <p class="small text-muted mt-2 mb-0"><strong>PyTorch Official Symbol</strong></p>
  </div>
  <div class="col-md-8">
    <h5 class="fw-bold text-dark mb-2">불꽃(Flame)과 연산 노드의 순환 궤도</h5>
    <p class="text-secondary small mb-3">
      파이토치의 공식 로고는 <strong>횃불(Torch)에서 피어오르는 붉은 불꽃</strong>과, 딥러닝 연산 그래프의 노드들이 유기적으로 연결되어 순환하는 <strong>아크(Arc) 궤도</strong>를 결합한 형태입니다.
    </p>
    <ul class="text-secondary small mb-3">
      <li><strong>타오르는 불꽃 (Flame)</strong>: 딥러닝과 인공지능이 인간의 삶과 기술을 밝히는 지적 탐구의 열정을 상징합니다.</li>
      <li><strong>원형 궤도와 화살표</strong>: 데이터 순전파(Forward)와 미분 역전파(Backward)가 끊임없이 반복되며 신경망이 스스로 최적화되는 순환 사이클을 의미합니다.</li>
      <li><strong>대표 컬러</strong>: 파이토치 오렌지-레드 (<span class="badge px-2 py-1 text-white" style="background-color: #EE4C2C;">#EE4C2C</span>)</li>
    </ul>
  </div>
</div>

---

## 4. 최신 파이토치 버전: PyTorch 2.x의 혁신

2023년 공개된 **PyTorch 2.0**과 최신 **PyTorch 2.5**는 파이토치 역사상 가장 거대한 패러다임 전환을 이뤄냈습니다:

| 핵심 기술 | 설명 | 효과 |
| :--- | :--- | :--- |
| **`torch.compile()`** | 파이썬 코드 한 줄 추가로 파이토치 코드를 고속 C++/Triton 커널로 자동 컴파일 | 별도 C++ 재작성 없이 **30%~200% 실행 속도 향상** |
| **TorchDynamo** | 파이썬 프레임 인터프리터 수준에서 계산 그래프를 안전하게 후킹하고 파싱 | 그래프 캡처 실패(Graph Break) 최소화 |
| **AOTAutograd** | 런타임 전에 미리 순전파와 역전파 계산 그래프를 미리 캡처 | 역전파 엔진의 극적인 메모리 및 속도 최적화 |
| **TorchInductor** | OpenAI의 GPU 언어인 Triton을 백엔드로 삼는 딥러닝 컴파일러 | 다양한 GPU 아키텍처에 최적화된 네이티브 코드 생성 |

---

## 5. 운영체제별 설치 가이드

파이토치는 운영체제와 보유하신 그래픽 카드(NVIDIA GPU, Apple Silicon Mac, CPU Only)에 따라 설치 명령어가 다릅니다.

{% include lecture_illustration.html 
   src="/assets/img/toto_maker_helper.jpg" 
   alt="조수 토토의 설치 환경 세팅 지원" 
   caption="자신의 하드웨어 사양에 꼭 맞는 파이토치를 설치해봅시다 — 조수 토토의 OS별 설치 도구 안내" 
   character="조수 토토" 
%}

{% include character_callout.html 
   char="toto" 
   title="운영체제별 탭을 확인하세요!" 
   text="멍멍! Mac(M1/M2/M3/M4)을 쓰시는 분은 MPS 지원 버전을, Windows/Linux에서 NVIDIA GPU를 쓰시는 분은 CUDA 버전을 설치하셔야 GPU 가속의 놀라운 속도를 경험할 수 있어요!" 
%}

### 🍏 1) macOS (Apple Silicon M1/M2/M3/M4 & Intel)
최신 macOS에서는 Apple이 독자 개발한 Metal GPU 가속 프레임워크인 **MPS(Metal Performance Shaders)** 백엔드를 지원합니다:

```bash
# 가상환경 활성화 후 최신 안정 버전 설치
pip3 install torch torchvision torchaudio
```

### 🪟 2) Windows (NVIDIA GPU / CPU)
Windows에서는 NVIDIA 그래픽 드라이버 버전과 호환되는 CUDA 버전을 선택해야 합니다:

```bash
# [NVIDIA GPU - CUDA 12.1 권장]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# [NVIDIA GPU - CUDA 11.8 구형 그래픽 카드]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# [CPU 전용 - 외장 그래픽 카드가 없는 경우]
pip3 install torch torchvision torchaudio
```

### 🐧 3) Linux (Ubuntu, Debian, RHEL)
리눅스 서버 및 워크스테이션 환경에서는 CUDA 또는 공식 도커 컨테이너를 추천합니다:

```bash
# [Linux CUDA 12.1]
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# [Docker 공식 컨테이너 실행]
docker run --gpus all -it --rm pytorch/pytorch:latest
```

---

## 6. 설치 검증 및 첫 텐서 테스트

터미널이나 Python REPL을 열고 아래 스크립트를 실행하여 파이토치가 올바르게 설치되었는지 확인합니다:

```python
import torch

print("📌 PyTorch Version:", torch.__version__)

# 하드웨어 가속 장치 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 NVIDIA GPU (CUDA) 가속 활성화됨:", torch.cuda.get_device_name(0))
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Apple Silicon GPU (MPS) 가속 활성화됨")
else:
    device = torch.device("cpu")
    print("💻 CPU 모드로 동작 중")

# 첫 텐서 생성 및 연산 테스트
x = torch.rand(3, 3, device=device)
y = torch.ones(3, 3, device=device)
z = x + y

print("\n[연산 결과 텐서 z]:\n", z)
```

{% include character_callout.html 
   char="tinny" 
   title="하드웨어 가속 연산 확인 완료!" 
   text="z 텐서 연산이 정상적으로 수행되었다면 여러분의 컴퓨터가 딥러닝 모델을 훈련시킬 완벽한 상태가 된 것입니다!" 
%}

---

## 🧭 다음 학습 단계로 나아가기

파이토치 설치와 환경 구성을 마쳤다면, 이제 본격적인 딥러닝 탐험을 시작할 차례입니다:

- **[01장. 딥러닝 시작](../01_딥러닝_시작/)**: 인공지능, 머신러닝, 딥러닝의 포함 관계와 인공 신경망의 역사를 배웁니다.
- **[02장. 파이토치를 위한 파이썬 선수 지식](../02_파이썬/)**: 파이썬 기초 마스터 클래스 연동 및 텐서와 넘파이 호환 원리를 확인합니다.
- **[04장. 파이토치 기본 (PyTorch Basics)](../04_파이토치_기본/)**: 텐서 생성, 조작, 슬라이싱, 자동 미분(Autograd)의 기본기를 다집니다.
- **[전체 커리큘럼](../curriculum/)**: 13개 챕터에 걸친 딥러닝 마스터 로드맵을 확인합니다.
