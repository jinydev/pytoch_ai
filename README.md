# 🔥 딥러닝을 위한 파이토치 입문 (PyTorch Basic)

<p align="center">
  <a href="https://pytoch.ai.jiny.dev">
    <img src="https://img.shields.io/badge/Site-pytoch.ai.jiny.dev-EE4C2C?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Live Site" />
  </a>
  <a href="https://github.com/jinydev/pytoch_ai">
    <img src="https://img.shields.io/badge/GitHub-jinydev%2Fpytoch__ai-black?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo" />
  </a>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Jekyll-4.4.1-CC0000?style=for-the-badge&logo=jekyll&logoColor=white" alt="Jekyll 4.4" />
</p>

---

## 💡 프로젝트 소개

**『딥러닝을 위한 파이토치 입문』**은 파이토치(PyTorch) 프레임워크를 기반으로 텐서(Tensor) 연산과 Autograd 자동 미분 원리부터 인공 신경망(MLP), CNN, RNN, 비지도 학습(GAN), 메타 학습, 과학적 계산(PINN)까지 단계별로 체계적으로 마스터하는 오픈 강의 교재 웹사이트입니다.

- 🌐 **공식 온라인 강의 사이트**: [https://pytoch.ai.jiny.dev](https://pytoch.ai.jiny.dev)
- 🖥️ **로컬 개발 서버**: `http://localhost:4002`

---

## 🔗 파이썬 기초 강좌 통합 안내

본 강좌에 포함되어 있던 기본 파이썬 문법(변수, 자료형, 함수, 객체지향 OOP, 모듈 등)은 더욱 깊이 있고 체계적인 48주 완성 전문 포털인 **[파이썬 마스터 클래스 (python.jiny.dev)](https://python.jiny.dev)**로 통합되었습니다.
중복 학습 없이 컴퓨터 과학 원리부터 시작하는 파이썬 완벽 마스터를 추천합니다.

- 🐍 **파이썬 마스터 클래스**: [https://python.jiny.dev](https://python.jiny.dev)
- 📌 **파이토치 2장 선수 가이드**: [`/02_파이썬/`](https://pytoch.ai.jiny.dev/02_파이썬/)

---

## 🗺️ 전체 커리큘럼 구성 (Curriculum)

| 장 | 구분 | 핵심 내용 |
| :---: | :--- | :--- |
| **소개** | **파이토치 소개 (`/intro/`)** | 파이토치 철학, Define-by-Run, 탄생사, 로고 상징, PyTorch 2.x, OS별(macOS MPS, Windows CUDA, Linux) 설치 |
| **01장** | **딥러닝 시작** | 인공지능/머신러닝/딥러닝 포함 관계, 인공 신경망의 역사 |
| **02장** | **파이토치 선수 지식** | 파이썬 마스터 클래스([python.jiny.dev](https://python.jiny.dev)) 연동 안내 및 넘파이-텐서 호환 |
| **03장** | **지도 학습** | 지도 학습의 분류와 회귀, 데이터 세트 분할(Train/Val/Test) |
| **04장** | **파이토치 기본** | 텐서 조작, Autograd 자동 미분, Dataset & DataLoader 미니배치 |
| **05장** | **인공 신경망 (ANN)** | 다층 퍼셉트론(MLP), 활성화 함수, 손실 함수, 옵티마이저, 교차 검증 |
| **06장** | **합성곱 신경망 (CNN)** | 합성곱/풀링 연산, AlexNet, ResNet(Skip Connection), VGG |
| **07장** | **순환 신경망 (RNN)** | 시계열 데이터 처리, Vanilla RNN, LSTM, GRU |
| **08장** | **비지도 학습** | K-평균 군집화, 오토인코더(Autoencoder), GAN, Style Transfer |
| **09장** | **성능 개선** | 드롭아웃, 데이터 증식(Augmentation), 전이 학습(Transfer Learning) |
| **10장** | **시각화** | 설명 가능한 AI(XAI, CAM/Grad-CAM), PCA, t-SNE 고차원 시각화 |
| **11장** | **메타 학습** | Learn to Learn, 퓨샷 러닝(Few-Shot Learning) |
| **12장** | **과학적 계산** | 물리 정보 기반 신경망 (PINN, Physics-Informed Neural Networks) |
| **13장** | **부록** | 참고 교재 정보 및 개발 환경 레퍼런스 |

---

## 💻 로컬 개발 서버 실행

```bash
# 의존성 설치
bundle install

# 핫리로드 로컬 개발 서버 실행 (포트 4002)
bundle exec jekyll serve --livereload --livereload-port 35731 --force_polling --port 4002

# 정적 사이트 배포 빌드 (docs/ 생성)
bundle exec jekyll build
```
