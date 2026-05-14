빠른 요약




# AI 가상 피팅 & 브랜드 룩북 생성기

> 

## 📌 프로젝트 개요

LoRA로 인물의 얼굴과 체형을 고정한 채, 다양한 의상을 자연스럽게 입혀볼 수 있는  
**AI 가상 피팅 자동화 파이프라인**입니다.

```
[인물 사진] → [YOLO/SAM 자동 마스킹] → [LLM 프롬프트 변환]
         → [SD Inpainting + LoRA] → [최종 룩북 이미지]
```

## 🛠️ 핵심 기술

| 기술 | 역할 |
|------|------|
| **LoRA** | 인물 얼굴/체형 일관성 유지 (15~20장 학습) |
| **Stable Diffusion Inpainting** | 의상 영역만 자연스럽게 교체 |
| **YOLOv8 + SAM** | 상의/하의 영역 자동 픽셀 분리 |
| **ControlNet** | 포즈 고정 (OpenPose/Canny) |
| **IP-Adapter** | 참조 이미지의 스타일·색감 적용 |
| **Gemini API** | 자연어 → SD 프롬프트 변환 |
| **Gradio** | 웹 데모 UI |

## 🚀 빠른 시작

### 1. 환경 설치

```bash
# scripts/install_windows.bat 더블클릭 또는:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정

```bash
# .env.example을 .env로 복사 후 키 입력
copy .env.example .env
# .env 파일에서 GEMINI_API_KEY 입력
```

### 3. 앱 실행

```bash
scripts\run_app.bat
# → http://localhost:7860 접속
```

### 4. LoRA 학습 (선택)

```bash
# data/input_faces/ 에 본인 사진 15~20장 넣은 후:
scripts\train_lora.bat
```

## 📁 프로젝트 구조

```
Fashion_Fitting/
├── src/
│   ├── masking.py          # YOLO + SAM 자동 마스킹
│   ├── inpainting.py       # SD Inpainting + LoRA
│   ├── prompt_generator.py # LLM 프롬프트 변환
│   └── pipeline.py         # 전체 파이프라인 통합
├── app/
│   └── app.py              # Gradio 웹 UI
├── models/                 # 모델 파일 (별도 다운로드)
├── data/                   # 학습/결과 이미지
└── scripts/                # 실행 스크립트
```

## 🎯 면접 포인트

1. **객체 고유성 보존**: LoRA로 인물의 얼굴을 매번 일관되게 유지 → 상업적 활용 가능
2. **자동 마스킹 파이프라인**: YOLO의 탐지력 + SAM의 정밀한 경계로 품질 향상
3. **LLM + Vision 결합**: 자연어 입력 → SD 프롬프트 자동 변환으로 UX 향상
4. **산업 적용 가능성**: 가구 배치(라이프스케이프)와 동일한 '객체 보존 + 배경 합성' 원리

## 📋 요구사항

- Python 3.10.x
- NVIDIA GPU (권장, VRAM 6GB 이상)
- Windows 10/11

## 📝 라이선스

포트폴리오 목적 프로젝트. 학습/연구용으로만 사용하세요.
