---
layout: docs
title: "5. 파이토치 운영체제별 설치 종합 가이드"
permalink: /intro/install.html
description: "macOS(Apple Silicon MPS), Windows(NVIDIA CUDA), Linux(CUDA/Docker) 등 운영체제별 최적화된 파이토치 설치 방법을 안내합니다."
keywords: "파이토치 설치, PyTorch install, pip install torch, PyTorch CUDA, PyTorch MPS, jinydev"
---

# 5. 파이토치 운영체제별 설치 종합 가이드

{% include lecture_illustration.html 
   src="/assets/img/toto_maker_helper.jpg" 
   alt="조수 토토의 설치 환경 세팅 지원" 
   caption="개발 환경을 올바르게 세팅하는 것이 딥러닝 여정의 절반입니다 — 조수 토토의 OS별 설치 가이드" 
   character="조수 토토" 
%}

파이토치는 공식 웹사이트([pytorch.org](https://pytorch.org))의 설치 매트릭스를 기반으로, 자신의 하드웨어(CPU vs GPU)와 운영체제(macOS, Windows, Linux)에 맞는 명령어를 선택하여 설치합니다.

---

## 🧭 빠른 운영체제별 바로가기

<div class="row g-3 my-4">
  <div class="col-md-4">
    <div class="card h-100 border-0 shadow-sm bg-light p-3 rounded-4">
      <h5 class="fw-bold text-dark">🍏 macOS 가이드</h5>
      <p class="small text-secondary mb-3">Apple Silicon(M1/M2/M3/M4)의 Metal(MPS) GPU 가속 설정</p>
      <a href="/intro/install-macos.html" class="btn btn-outline-dark btn-sm rounded-pill mt-auto">macOS 설치법 보기 ➔</a>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card h-100 border-0 shadow-sm bg-light p-3 rounded-4">
      <h5 class="fw-bold text-dark">🪟 Windows 가이드</h5>
      <p class="small text-secondary mb-3">NVIDIA CUDA 드라이버 및 Windows pip/conda 완벽 설치</p>
      <a href="/intro/install-windows.html" class="btn btn-outline-primary btn-sm rounded-pill mt-auto">Windows 설치법 보기 ➔</a>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card h-100 border-0 shadow-sm bg-light p-3 rounded-4">
      <h5 class="fw-bold text-dark">🐧 Linux 가이드</h5>
      <p class="small text-secondary mb-3">서버/클라우드 환경의 CUDA 12.x 및 공식 Docker 실행</p>
      <a href="/intro/install-linux.html" class="btn btn-outline-success btn-sm rounded-pill mt-auto">Linux 설치법 보기 ➔</a>
    </div>
  </div>
</div>

---

## 📦 Conda vs Pip: 무엇을 선택해야 할까?

| 구분 | Pip (`pip install torch`) | Conda (`conda install pytorch`) |
| :--- | :--- | :--- |
| **장점** | 설치가 가볍고 빠름, PyPI 최신 패키지와 즉시 호환 | 복잡한 NVIDIA CUDA 드라이버/cuDNN 툴킷을 알아서 함께 관리 |
| **추천 대상** | **대부분의 개발자, macOS 사용자, Docker 환경 (강력 권장)** | Windows에서 CUDA 툴킷 설정이 까다로운 초보자 |

{% include character_callout.html 
   char="jiny" 
   title="가상환경 격리는 필수냥!" 
   text="시스템 전역에 파이토치를 설치하면 다른 패키지 버전과 충돌이 발생한다냥! 반드시 python -m venv .venv 또는 conda create로 가상환경을 먼저 만들고 활성화한 뒤 설치하라냥!" 
%}
