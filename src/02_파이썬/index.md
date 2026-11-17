---
layout: docs
title: "02장 파이토치를 위한 파이썬 선수 지식 & 연동 가이드"
permalink: /02_파이썬/
---

# CHAPTER 2 파이토치를 위한 파이썬 선수 지식

파이토치(PyTorch)는 파이썬(Python) 기반의 오픈소스 딥러닝 프레임워크입니다. 따라서 딥러닝 모델을 설계하고 학습시키기 위해서는 파이썬에 대한 기본 지식과 객체지향 프로그래밍(OOP) 감각이 필수적입니다.

---

<div class="card my-4 border-primary border-2 shadow-sm rounded-4 overflow-hidden">
  <div class="card-header bg-primary text-white py-3">
    <h5 class="mb-0 fw-bold">🐍 파이썬 강좌 통합 및 전용 사이트 안내</h5>
  </div>
  <div class="card-body p-4 bg-light">
    <div class="row align-items-center">
      <div class="col-md-8">
        <h5 class="fw-bold text-dark mb-2">파이썬 기초는 <a href="https://python.jiny.dev" target="_blank" class="text-primary text-decoration-none">파이썬 마스터 클래스</a>에서 완벽하게 제공됩니다!</h5>
        <p class="text-secondary small mb-3">
          본 강좌에 포함되어 있던 기본 파이썬 문법(변수, 컨테이너 자료형, 제어문, 함수, 클래스, 모듈, 파일 I/O 등)은 
          더욱 깊이 있고 체계적인 48주 완성 전문 사이트인 <strong>파이썬 마스터 클래스 (python.jiny.dev)</strong>로 통합 이전되었습니다.
          중복 학습 없이 컴퓨터 과학(CS)의 본질적 시각에서 파이썬을 학습하실 수 있습니다.
        </p>
        <div class="d-flex gap-2 flex-wrap">
          <a href="https://python.jiny.dev" target="_blank" class="btn btn-primary rounded-pill px-4 fw-bold shadow-sm">
            파이썬 마스터 클래스 바로가기 ➔
          </a>
          <a href="https://python.jiny.dev/intro/" target="_blank" class="btn btn-outline-primary rounded-pill px-3">
            파이썬 언어 소개
          </a>
          <a href="https://python.jiny.dev/semester1/" target="_blank" class="btn btn-outline-primary rounded-pill px-3">
            1학기 기초 & CS 원리
          </a>
        </div>
      </div>
      <div class="col-md-4 text-center mt-3 mt-md-0">
        <img src="/assets/img/logos/logo_01_python.jpg" alt="파이썬 마스터 클래스" class="img-fluid rounded-4 shadow-sm" style="max-height: 180px; object-fit: cover;">
      </div>
    </div>
  </div>
</div>

---

## 2.1 파이토치 학습 전 필수 체크리스트

파이토치 딥러닝으로 넘어가기 전, 아래 항목들을 스스로 점검해 보세요:

1. **기본 자료형과 시퀀스 조작**:
   - `list`, `tuple`, `dict`, `set`의 차이와 적절한 사용 시점
   - 리스트 컴프리헨션(List Comprehension) 및 슬라이싱(`[start:stop:step]`)
2. **함수와 언패킹**:
   - 가변 위치 인수(`*args`)와 키워드 가변 인수(`**kwargs`)의 동작 방식
   - 람다(Lambda) 및 고차 함수
3. **객체지향 프로그래밍 (OOP)**:
   - 클래스 선언, 인스턴스 생성, `__init__` 생성자
   - 상속(`super().__init__()`)과 메서드 오버라이딩 (PyTorch의 `nn.Module` 상속의 기초)
   - 객체를 함수처럼 호출할 수 있게 만드는 `__call__` 매직 메서드
4. **수치 연산 라이브러리 (NumPy)**:
   - `ndarray` 배열 생성, 차원(`ndim`, `shape`), 데이터 타입(`dtype`)
   - 행렬 곱(`@`, `np.dot`), 브로드캐스팅(Broadcasting)

---

## 2.2 파이토치 텐서(Tensor)와 넘파이(NumPy)의 상호 호환

파이토치 텐서는 넘파이의 다차원 배열 `ndarray`와 매우 유사하며, 메모리를 공유하면서 효율적으로 상호 변환됩니다.

### 1) NumPy 배열을 PyTorch 텐서로 변환 (`from_numpy`)
```python
import numpy as np
import torch

# NumPy ndarray 생성
np_arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# PyTorch 텐서로 변환 (메모리 버퍼를 공유)
tensor_from_np = torch.from_numpy(np_arr)
print("PyTorch Tensor:\n", tensor_from_np)
```

### 2) PyTorch 텐서를 NumPy 배열로 변환 (`numpy()`)
```python
# 텐서를 넘파이로 다시 변환
back_to_np = tensor_from_np.numpy()
print("NumPy Array:\n", back_to_np)
```

> [!WARNING]
> GPU 텐서(`cuda` 또는 `mps`)는 바로 `.numpy()`를 호출할 수 없습니다.  
> 반드시 CPU 메모리로 복사한 후 변환해야 합니다: `tensor.detach().cpu().numpy()`

---

## 2.3 딥러닝 구현 시 빈출되는 핵심 파이썬 관용구

### 1) `nn.Module` 상속과 `super().__init__()`
파이토치에서 신경망 계층이나 모델을 정의할 때는 반드시 `nn.Module`을 상속받습니다:

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()  # 부모 클래스(nn.Module)의 초기화 루틴 실행
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 순전파(Forward pass) 연산 정의
        return self.relu(self.linear(x))
```

### 2) `__call__` 매직 메서드와 인스턴스 실행
`model(x)` 형태로 인스턴스를 마치 함수처럼 호출하면, 파이토치는 내부적으로 훅(Hook)을 실행한 뒤 사용자가 정의한 `forward(x)` 메서드를 호출합니다.

---

## 🧭 다음 학습 단계

파이썬의 핵심 연동 원리를 확인하셨다면, 이제 본격적인 파이토치 프레임워크 학습으로 진행합니다:

- **[03장 지도 학습 (Supervised Learning)](../03_지도_학습/)**: 머신러닝/딥러닝의 가장 핵심적인 지도 학습 분류와 회귀
- **[04장 파이토치 기본 (PyTorch Basics)](../04_파이토치_기본/)**: 텐서 연산, Autograd 자동 미분 엔진, DataLoader 구축
