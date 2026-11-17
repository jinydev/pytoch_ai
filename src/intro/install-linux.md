---
layout: docs
title: "🐧 Linux 파이토치 설치 가이드 (CUDA, ROCm, Docker)"
permalink: /intro/install-linux.html
description: "Ubuntu, Debian 등 Linux 서버 및 워크스테이션에서 최신 CUDA 환경과 Docker 컨테이너로 파이토치를 구동하는 방법을 안내합니다."
keywords: "PyTorch Linux, Ubuntu PyTorch, PyTorch Docker, Linux CUDA PyTorch, jinydev"
---

# 🐧 Linux 파이토치 설치 가이드 (CUDA, ROCm, Docker)

리눅스는 대부분의 딥러닝 연구소, 클라우드 인스턴스(AWS EC2, GCP, Azure), GPU 클러스터 서버에서 사용하는 표준 운영체제입니다.

---

## 🛠️ 1. Ubuntu / Debian 표준 Pip 설치

```bash
# 1. 필수 시스템 패키지 설치
sudo apt update && sudo apt install -y python3-pip python3-venv build-essential

# 2. 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 3. 최신 CUDA 12.1용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🐳 2. 공식 Docker 컨테이너 실행 (가장 추천)

호스트 OS에 복잡한 CUDA Toolkit과 라이브러리를 일일이 설치하지 않고도, NVIDIA Container Toolkit만 설치되어 있다면 단 한 줄로 완벽히 격리된 파이토치 딥러닝 환경을 띄울 수 있습니다:

```bash
# 대화형 bash 쉘로 최신 파이토치 컨테이너 진입
docker run --gpus all -it --rm --shm-size=16g pytorch/pytorch:latest
```

> [!TIP]
> **공유 메모리 설정 (`--shm-size=16g`)**:  
> 파이토치의 `DataLoader`는 멀티프로세싱(`num_workers > 0`)으로 데이터를 비동기 로딩할 때 리눅스 공유 메모리(`/dev/shm`)를 대량으로 사용합니다.  
> 도커 기본값(64MB)을 그대로 두면 `Bus error`가 발생하므로 반드시 `--shm-size`를 8G 이상 넉넉히 설정해야 합니다.
