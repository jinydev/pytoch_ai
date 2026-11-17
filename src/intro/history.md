---
layout: docs
title: "2. 파이토치 탄생과 발전 배경 (History)"
permalink: /intro/history.html
description: "Lua Torch부터 Chainer의 영감, FAIR의 오픈소스 공개, 리눅스 재단 PyTorch Foundation 이관까지 파이토치의 발전사를 살펴봅니다."
keywords: "파이토치 역사, Torch, Lua, FAIR, Soumith Chintala, PyTorch Foundation, jinydev"
---

# 2. 파이토치 탄생과 발전 배경 (History)

{% include lecture_illustration.html 
   src="/assets/img/hojin_lab_lecture.jpg" 
   alt="이호진 교수님의 딥러닝 프레임워크 역사 특강" 
   caption="학계의 열정과 오픈소스 커뮤니티의 연대가 만들어낸 딥러닝 혁명 — 이호진 교수님의 파이토치 역사 특강" 
   character="이호진 교수님" 
%}

파이토치(PyTorch)는 어느 날 갑자기 하늘에서 떨어진 기술이 아닙니다.  
2000년대 초반 과학 연산 라이브러리부터 시작하여, 일본 스타트업의 혁신적인 아이디어와 페이스북 인공지능 연구소(FAIR)의 공학적 실행력이 결합되어 탄생했습니다.

---

## 📜 파이토치 발전 연표 (Timeline)

| 연도 | 주요 사건 | 의미 |
| :---: | :--- | :--- |
| **2002년** | **Torch 프로젝트 시작** (Ronan Collobert 등) | Lua 언어 기반의 과학 연산 및 머신러닝 라이브러리로 시작 |
| **2015년** | **Chainer 발표** (Preferred Networks) | 세계 최초로 **Define-by-Run(동적 계산 그래프)** 개념을 제안하여 학계에 신선한 충격을 줌 |
| **2016년** | **PyTorch 개발 착수** (FAIR) | 아담 파스케(Adam Paszke), 수미스 친탈라(Soumith Chintala) 등이 Chainer의 동적 그래프 철학과 Torch의 고성능 C 백엔드를 융합하여 파이썬 네이티브 프레임워크 착수 |
| **2017년 1월** | **PyTorch v0.1 오픈소스 최초 공개** | 간결하고 직관적인 문법으로 딥러닝 연구자들의 폭발적인 호응 획득 |
| **2018년 12월** | **PyTorch 1.0 릴리스** | Caffe2와 완전히 병합, TorchScript(JIT)와 C++ 서빙(LibTorch) 지원으로 연구와 상용 배포의 통합 달성 |
| **2022년 9월** | **PyTorch Foundation 설립** | Meta에서 **리눅스 재단(Linux Foundation)** 산하의 중립적 독립 기구로 거버넌스 공식 이관 |
| **2023년 3월** | **PyTorch 2.0 릴리스** | `torch.compile()`, TorchDynamo, TorchInductor 도입으로 파이썬 네이티브 컴파일러 시대 개막 |

---

## 🏛️ 거버넌스의 독립: PyTorch Foundation

2022년 9월, 메타(Meta)의 마크 저커버그는 파이토치를 **리눅스 재단(Linux Foundation)** 산하의 비영리 오픈소스 프로젝트로 이관한다고 발표했습니다.

- **창립 이사회 멤버**: Meta, AMD, AWS, Google Cloud, Microsoft, NVIDIA 등 글로벌 빅테크 연합
- **목적**: 특정 단일 기업의 이해관계에 종속되지 않고, 하드웨어 중립적이며 학계와 산업계 누구나 자유롭게 기여할 수 있는 공공 소프트웨어 인프라로의 진화

{% include character_callout.html 
   char="hojin" 
   title="오픈소스 생태계의 승리" 
   text="파이토치가 리눅스 재단으로 이관된 것은 리눅스(Linux)나 쿠버네티스(Kubernetes)처럼 인류 전체가 공유하는 핵심 인공지능 디지털 공공재로 인정받았음을 의미한단다." 
%}
