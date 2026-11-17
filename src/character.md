---
layout: docs
title: "캐릭터 소개 및 세계관 가이드"
permalink: /character/
---

# 🧭 딥러닝 프레임워크 탐험대: 캐릭터 소개 및 세계관

이 강의 노트는 **호기심 넘치는 주니어 메이커 도로시**, 도로시의 단짝 반려견 **토토**, 이호진 교수님의 수석 연구실 **조교 지니**, 그리고 지적이고 친근한 **이호진 교수님**이 함께 딥러닝 프레임워크를 밑바닥부터 구축해 나가는 모험 이야기를 다룹니다.

딱딱하고 추상적인 **딥러닝 프레임워크(DeZero / PyTorch 아키텍처)**의 동작 원리와 저수준 수학적 알고리즘을 스튜디오 지브리 감성의 따뜻한 스토리텔링으로 풀어내어, 초보자부터 전공자까지 누구나 즐겁게 프레임워크의 본질을 정복할 수 있도록 안내합니다.

---

## 👥 등장인물 소개 (The Characters)

딥러닝 프레임워크의 5개 고지(60단계)를 함께 등반하는 4명의 핵심 주인공들입니다.

### 👨‍🏫 이호진 교수님 (Prof. Hojin)
* **역할 / 페르소나**: **지적이고 친근한 AI/공학 멘토 (스티브 잡스 스타일의 실용적 혁신가)**
* **캐릭터 이미지**:
  <div class="my-3 text-center text-sm-start">
    <img src="/img/hojin_profile.jpg" alt="이호진 교수님" style="max-height: 220px; width: auto; object-fit: contain;">
  </div>
* **외형 및 시각적 특징 (Visual Specs)**:
  - 40대 초반의 생기 있고 스마트한 동안 (`early 40s, youthful face, smooth skin without wrinkles`)
  - 인자하고 총명한 갈색 눈, 학생들을 북돋아 주는 따뜻하고 자신감 넘치는 미소
  - 단정하게 가르마를 탄 솔트앤페퍼 숏헤어 (다크 차콜 흑발에 자연스러운 은빛 솔트앤페퍼)
  - 얇은 실버 라운드 메탈 와이어 안경
  - 단정한 **다크 차콜 네이비 단색 카라 반팔 폴로 셔츠** (이름표나 로고 없음), 클래식 중청 스트레이트 청바지, 뉴발란스 그레이 스웨이드 스니커즈
* **DeZero 프레임워크 속 역할**:
  - 개념의 원리와 설계 흐름을 명쾌하게 짚어주는 나침반 역할을 담당합니다.
  - *"왜 넘파이 배열만으로는 역전파를 자동화할 수 없을까?"*, *"Define-by-Run은 런타임에 어떻게 그래프를 엮어낼까?"*와 같은 핵심 화두를 던져 수식과 아키텍처의 길을 열어줍니다.

---

### 🐱 조교 지니 (Jiny)
* **역할 / 페르소나**: **이호진 교수님의 수석 연구실 조교 (시크하면서도 똑똑한 츤데레 냥이)**
* **캐릭터 이미지**:
  <div class="my-3 text-center text-sm-start">
    <img src="/img/jiny_profile.jpg" alt="조교 지니" style="max-height: 220px; width: auto; object-fit: contain;">
  </div>
* **외형 및 시각적 특징 (Visual Specs)**:
  - 한국 토종 코리안 숏헤어 젖소/턱시도 바이컬러 (검은 마스크, 흰 이마 가르마선, 순백의 가슴/배/네 발, 등과 옆구리의 블랙 패치)
  - 지적이고 맑은 **올리브 그린 눈동자**
  - 사랑스러운 **베이비 핑크 단추 코**
  - 코끝에 살짝 걸친 **미니 원형 조교 안경**
  - 작은 황금 방울이 달린 선명한 **빨간 가죽 목걸이**
* **DeZero 프레임워크 속 역할**:
  - 연구실 모니터 앞이나 칠판 위에서 꼬리를 흔들며 코드 디버깅을 감수합니다.
  - 도로시가 작성한 순전파·역전파 코드에서 변수와 함수가 올바르게 참조 연결되었는지 앞발로 콕 짚어 검증해 줍니다.
  - 복잡한 미분 수식을 직관적인 **계산 그래프(DAG)**로 시각화하여 보여줍니다.

---

### 👧 도로시 (Dorothy)
* **역할 / 페르소나**: **호기심 넘치는 주니어 메이커 (실패를 두려워하지 않는 당찬 소녀 코더)**
* **캐릭터 이미지**:
  <div class="my-3 text-center text-sm-start">
    <img src="/img/dorothy_profile.png" alt="도로시" style="max-height: 220px; width: auto; object-fit: contain;">
  </div>
* **외형 및 시각적 특징 (Visual Specs)**:
  - **4.5등신 치비/주니어 비율**의 사랑스러운 9~10세 소녀
  - 볼륨감 있는 밤갈색 웨이브 양갈래 머리 (알록달록 구슬 방울 머리끈)와 이마를 덮는 가지런한 앞머리
  - 반짝이는 따뜻한 호박 갈색(Warm Amber) 눈망울과 발그레한 피치 핑크 홍조
  - 클래식 미디엄 블루 데님 멜빵바지 (가슴 중앙에 원형 **'STEM HERO'** 패치)
  - 안쪽에 받쳐 입은 주황/노랑 가로 줄무늬 반팔 티셔츠, 흰 양말, **비비드 레드 캔버스 스니커즈**
* **DeZero 프레임워크 속 역할**:
  - DeZero 프레임워크를 1단계부터 60단계까지 한 줄 한 줄 직접 파이썬 코드로 구현해 나가는 주인공입니다.
  - 상자로서의 `Variable`, 변환 장치인 `Function`, 연산자 오버로드, 메모리 관리, 고차 미분, 신경망 계층(`Layer`), 최적화(`Optimizer`)를 차근차근 조립하며 강력한 딥러닝 엔진을 완성합니다.

---

### 🐶 토토 (Toto)
* **역할 / 페르소나**: **도로시의 단짝 반려견 & 마스코트 (지치지 않는 비타민이자 현장 조수)**
* **캐릭터 이미지**:
  <div class="my-3 text-center text-sm-start">
    <img src="/img/toto_profile.jpg" alt="토토" style="max-height: 220px; width: auto; object-fit: contain;">
  </div>
* **외형 및 시각적 특징 (Visual Specs)**:
  - 비숑 프리제와 말티즈 믹스 아기 강아지
  - 반점과 얼룩이 전혀 없는 **단색 웜 밀크 화이트 / 크림 아이보리 (`#FBF6EB`)** 구름 같은 소프트 웨이브 털
  - 뺨을 감싸는 둥근 드룹 귀와 동그란 흑진주 눈동자
  - 반질반질 윤기 나는 **검은 단추 코 (블랙 버튼 노즈)**
  - 핑크빛 혀를 살짝 내민 해맑은 미소
  - 목에 산뜻하게 묶은 **스카이 블루 네커치프 삼각 반다나**
* **DeZero 프레임워크 속 역할**:
  - 도로시의 곁을 맴돌며 무한한 응원과 활력을 주는 마스코트입니다.
  - 텐서 연산 시 차원 불일치(Shape Error)나 브로드캐스팅 오류, 순환 참조로 인한 메모리 누수의 냄새를 기가 막히게 맡아내어 멍멍 짖으며 알려줍니다.

---

## 🏗️ DeZero 프레임워크 탐험대의 역할 매트릭스

```
┌────────────────────────────────────────────────────────┐
│               [이호진 교수님 (Prof. Hojin)]            │
│               • 수학적 알고리즘 & 설계 철학            │
│               • Autograd 엔진 나침반                   │
└───────────────────────────┬────────────────────────────┘
                            │ (원리 전수 & 지도)
                            ▼
┌────────────────────────────────────────────────────────┐
│                  [조교 지니 (Jiny)]                    │
│                  • 계산 그래프(DAG) 시각화             │
│                  • 세대 기반 역전파 디버깅 감수        │
└───────────────────────────┬────────────────────────────┘
                            │ (디버깅 지원 & 힌트)
                            ▼
┌───────────────────────────────────────┐   ┌────────────┐
│          [도로시 (Dorothy)]           │◄──┤[토토(Toto)]│
│  • Variable & Function 클래스 구현    │   │ • 버그 탐지│
│  • 연산자 오버로드 & 고차 미분 구현   │   │ • 형상 오류│
│  • Layer, Optimizer, CNN, RNN 신경망  │   │   체크     │
└───────────────────┬───────────────────┘   └────────────┘
                    │ (파이썬 코드로 완성)
                    ▼
┌────────────────────────────────────────────────────────┐
│             DeZero 딥러닝 프레임워크 (60단계)          │
└────────────────────────────────────────────────────────┘
```

---

## 🎨 일러스트 생성 가이드 (Official Prompt Formula)

도로시 세계관의 모든 이미지는 **스튜디오 지브리 2D 애니메이션 스타일**의 일관성을 유지해야 합니다. 
새로운 상황이나 개념 삽화를 생성할 때는 [`dorothy/prompt.md`](file:///Users/hojin9/Library/Mobile%20Documents/com~apple~CloudDocs/02.강의/dorothy/prompt.md)의 5블록 아키텍처를 준수합니다:

### 1. 절대적 3대 규칙 (Golden Constraints)
1. **NO 테두리/프레임**: 발표 슬라이드 테두리, 카드 박스, 사각/원형 프레임 절대 금지 (순수 피사체만 출력).
2. **순백색 배경 (`#FFFFFF`)**: 투명 PNG처럼 자연스럽게 녹아들도록 무결점 순백색 배경과 발밑 소프트 그림자만 허용.
3. **지브리 2D 애니메이션 화풍**: 다크 에스프레소 브라운 라인 아트(`#26211E`), 따뜻한 플랫 셀 셰이딩 (3D 사실화 금지).

### 2. 캐릭터별 표준 프롬프트 스니펫
* **이호진 교수님**:
  ```text
  Professor Hojin, a youthful and friendly Korean engineering professor in his early 40s (smooth skin, no wrinkles, energetic kind face), neatly parted stylish salt-and-pepper short black hair, thin minimalist round metal wire-rim glasses, kind intelligent warm brown eyes, encouraging warm smile. Dressed in a neat solid dark charcoal navy short-sleeve collared polo shirt, classic blue straight denim jeans, grey retro sneakers
  ```
* **조교 지니**:
  ```text
  Jiny the lab assistant cat, an adorable slightly chubby black-and-white bicolor tuxedo cat with cow pattern fur, black mask over ears, pure white muzzle with baby pink button nose, white chest and white paws, intelligent olive-green eyes, wearing a charming red collar with tiny gold bell and miniature round wire-rim scholar glasses on nose
  ```
* **도로시**:
  ```text
  Dorothy, an adorable 9-year-old anime girl, 4.5 heads tall junior proportions, wavy chestnut brown hair in twin pigtails with colorful round beads, cute bangs, warm amber-brown eyes, rosy cheeks, cheerful smile. Wearing classic blue denim overalls with rolled cuffs and a 'STEM HERO' patch on chest, orange and yellow striped t-shirt underneath, vibrant ruby red canvas sneakers
  ```
* **토토**:
  ```text
  Toto the puppy, an adorable fluffy cream-white puppy (bichon-maltese mix) with a uniform solid warm milk-white coat and NO spots, cloud-like soft wavy fur, soft floppy ears, big round black button eyes, glossy black button nose, cheerful dog smile, wearing a vibrant sky-blue neckerchief bandana
  ```
